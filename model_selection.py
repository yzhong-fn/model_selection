from pineapple.contrib.loaders.experiment_loader import ExperimentLoader, InferenceLoader
from pineapple.contrib.results.results_loader import load_results
import seaborn as sns
import matplotlib.pyplot as plt 
import numpy as np
from scipy.stats import ttest_ind, ttest_1samp, wilcoxon, ttest_rel, mannwhitneyu
from scipy import stats
import collections
import pandas as pd
from operator import itemgetter, attrgetter
import helper
import pickle
from scipy import stats


def get_CV_performance(bucket, model_path, metric):
    train = ExperimentLoader(train_bucket = bucket, train_path = model_path)
    train_perf = train.aggregated_fold_metric[f'test_{metric}_mean']
    return train_perf

def get_age_correlation_with_model_score(bucket, model_path):
    res = load_results(bucket = bucket, path = model_path)
    table_score = res[2]
    table_score['age'] = [m.age_at_blood_draw for m in table_score['metadata']]
    table_score['score0'] = table_score.score.apply(lambda x: x[0])
    table_score['score1'] = table_score.score.apply(lambda x: x[1])
    table_score['score1_mean'] = table_score.groupby(['sample_id'])['score1'].transform(lambda x: np.mean(x))
    table_score = table_score[['sample_id', 'score1_mean', 'label', 'age']].drop_duplicates()
    ## calculate pearson correlation
    cor_test = stats.pearsonr(table_score['age'], table_score['score1_mean'])
    return cor_test
    
def collect_cv_metric(bucket, model_path, metric_name):
    loader = ExperimentLoader(
            train_bucket=bucket,
            train_path=model_path,
            inference_bucket=None,
            inference_path=None,
        )
    
    metric = np.array([
            loader.fold(fold_name).model_fold_metric["test_" + metric_name]
            for fold_name in loader.fold_names()
        ])
    return metric

    
def signed_rank_test_direction(x, y):
    '''
    get the direction of signed rank test
    if the sum of positive ranks larger than the sum of negative ranks, return true; ote
    '''
    d = x - y
    d = np.compress(np.not_equal(d, 0), d)
    r = stats.rankdata(abs(d))
    r_plus = np.sum((d > 0) * r)
    r_minus = np.sum((d < 0) * r)
    return r_plus > r_minus


class model_selection():
    
    def __init__(self, buckets, model_paths):
        self.models = model_paths
        self.buckets = buckets    
        self.n = len(model_paths)
        self.performance_metrics = {}
        self.confounding_metrics = {}
        self.pairwise_model_comparison_results = []
        self.first_tier_models = []
        self.paired_comparison = collections.defaultdict(dict)
        
    def calculate_metrics(self):
        print('calculating performance metrics...')
        self.performance_metrics = self.get_performance_metrics()
        print('calculating confounding metrics...')
        self.confounding_metrics = self.get_confounding_metrics()
        print('calculating paired model comparison...')
        self.pairwise_model_comparison_results = self.paired_model_comparison(self.n)
        print('selecting first tier models...')
        self.first_tier_models = self.select_first_tier_models(self.pairwise_model_comparison_results)
        return 
    
    def paired_model_comparison(self, n):
        #do paired non parametric test
        sens90specs = [
        collect_cv_metric(self.buckets[i], self.models[i], "sens_upperthresh_spec90") for i in range(n)
        ]
        self.cv_metrics = {self.models[i]: sens90specs[i] for i in range(len(self.models))}
        self.cv_std = {self.models[i]: np.std(sens90specs[i]) for i in range(len(self.models))} #standard deviation
        self.cv_sem = {self.models[i]: stats.sem(sens90specs[i]) for i in range(len(self.models))} #standard error
        paired_test_results = []
        for i in range(n):
            for j in range(i + 1, n):
                w, p = wilcoxon(sens90specs[i], sens90specs[j])
                direction = signed_rank_test_direction(sens90specs[i], sens90specs[j])
                paired_test_results.append([self.models[i], self.models[j], p, direction])
                self.paired_comparison[self.models[i]][self.models[j]] = [self.models[i], self.models[j], p, direction] #for quickly retrieve the paired test results
                self.paired_comparison[self.models[j]][self.models[i]] = [self.models[i], self.models[j], p, direction]
        
        return paired_test_results
    
    
    def get_performance_metrics(self, metric='sens_upperthresh_spec90'):
        for i in range(self.n):
            self.performance_metrics[self.models[i]] = get_CV_performance(self.buckets[i], self.models[i], metric)
        return self.performance_metrics
    
    
    def get_confounding_metrics(self):
        for i in range(self.n):
            self.confounding_metrics[self.models[i]] = get_age_correlation_with_model_score(self.buckets[i], self.models[i])[0]
        return self.confounding_metrics
    
    
    def select_first_tier_models(self, paired_model_tests: list):
        '''
        select models that are not significantly worse than any other models
        '''
        in_bucket = set()
        out_bucket = set()
        for test in paired_model_tests:
            m1, m2, p, greater = test[0], test[1], test[2], test[3]
            if p < 0.05: #paired model test is significant
                if greater:
                    better_model = m1
                    worse_model = m2
                else:
                    better_model = m2
                    worse_model = m1
                out_bucket.add(worse_model)
                if worse_model in in_bucket:
                    in_bucket.remove(worse_model)
                if better_model not in out_bucket:
                    in_bucket.add(better_model)
            else: #paired model test is not significant
                if m1 not in out_bucket:
                    in_bucket.add(m1)
                if m2 not in out_bucket:
                    in_bucket.add(m2)
            self.in_bucket = in_bucket
            self.out_bucket = out_bucket
        return list(self.in_bucket)
    
    
    def rank_model_by_metrics(self, models, metric_dict):
        metrics = [metric_dict[m] for m in models]
        # rank model by performance metrics
        return [x for x, _ in sorted(zip(models, metrics), key=lambda pair: pair[1], reverse=True)] #decreasing order
    
    def rank_model_by_complexity(self, best_model, complexity_rank_dict, metric_dict):
        self.complexity_ranks = complexity_rank_dict
        # rank model by complexity and One-standard-error rule
        # assume the larger the metrics, the better the model
        lower_bound = self.performance_metrics[best_model] - self.cv_sem[best_model]
        models_within_OSE = list(np.array(self.first_tier_models)[[self.performance_metrics[x] >= lower_bound for x in self.first_tier_models]])
        metrics = [metric_dict[m] for m in models_within_OSE]
        ranks = [complexity_rank_dict[m] for m in models_within_OSE]
        return [x for x, _, _ in sorted(zip(models_within_OSE, ranks, metrics), key=itemgetter(1,2), reverse=True)] #both decreasing order
        