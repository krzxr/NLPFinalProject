from result_analysis import *
'''
datasets = ['./datasets/shuffle_unigrams_2authors/shuffle_unigrams_2authors_t80t10v10.pkl',
            './datasets/old_data/train_80_test_10_valid_top_2.pkl',
            './datasets/double_matches_2authors/double_matches_2authors_t80t10v10.pkl',
            './datasets/double_matches_lowercase_2authors/double_matches_lowercase_2authors_t80t10v10.pkl',
            './datasets/double_matches_unigram_2authors/double_matches_unigram_2authors_t80t10v10.pkl',
            './datasets/lowercase_2authors/lowercase_2authors_t80t10v10.pkl',
            './datasets/same_num_blogs_2authors/same_num_blogs_2authors_t80t10v10.pkl',
            './datasets/single_normal_2authors/single_normal_2authors_t80t10v10.pkl']


models = ['shuffle_unigrams_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'old_result_2_authors/checkpoint-3000',
          'single_normal_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760']
'''

datasets = {'same-word':'./datasets/same_num_shuffle_2authors/same_num_shuffle_2authors_t80t10v10.pkl',
            'same-lower':'./datasets/same_num_lowercase_2authors/same_num_lowercase_2authors_t80t10v10.pkl',
            'normal':'./datasets/single_normal_2authors/single_normal_2authors_t80t10v10.pkl',
            'double':'./datasets/double_matches_2authors/double_matches_2authors_t80t10v10.pkl',
            'double_lower':'./datasets/double_matches_lowercase_2authors/double_matches_lowercase_2authors_t80t10v10.pkl',
            'double_unigram':'./datasets/double_matches_unigram_2authors/double_matches_unigram_2authors_t80t10v10.pkl',
            'same_blog':'./datasets/same_num_blogs_2authors/same_num_blogs_2authors_t80t10v10.pkl',
            'same-no-pun':'./datasets/same_num_no_pun_2authors/same_num_no_pun_2authors_t80t10v10.pkl'}

models = {'word':'shuffle_unigrams_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'normal':'single_normal_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'lower': 'lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-12760',
          'double': 'double_matches_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660',
          'double_lower': 'double_matches_lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660',
          'double_unigram':'double_matches_unigram_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8227',
          'same_blog':'same_num_blogs_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660',
          'same_shuffle':'same_num_shuffle_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660/',
          'same_lowercase':'same_num_lowercase_2authors_attempt_4_epochs_20_lr_0.005/checkpoint-8660/',
          'same_no_pun':'same_num_no_pun_2authors_attempt_5_epochs_20_lr_0.005/checkpoint-8660'}
# word shuffled model on shuffled dataset
normal='normal_model_on_normal'
shuffle='shuffle_model_on_normal'
lowercase = 'lowercase_model_on_normal'
same_blog = "same_blog_on_normal"
double_on_double = "double_match_model_on_dobule_match"
double_lower_on_double = "double_match_lowercase_model_on_dobule_match"
double_unigram_on_double = "double_match_unigram_model_on_dobule_match"
double_on_same = "double_match_model_on_same_blog"
same_on_same = "same_blog_model_on_same_blog"

'''
error_detection('normal_model_on_normal/',input_file=datasets['normal'], name=models['normal'])
error_detection('shuffle_model_on_normal/',input_file=datasets['normal'], name=models['word'])
error_detection('lowercase_model_on_normal/',input_file=datasets['normal'], name=models['lower'])
'''
# error_detection(same_blog+'/',input_file=datasets['normal'], name=models['same_blog'])

# error_detection('same_blog_model_on_double_match/',input_file=datasets['double'], name=models['same_blog'])
# error_detection('double_match_model_on_same_blog/',input_file=datasets['same_blog'], name=models['double'])
# error_detection('double_match_model_on_dobule_match/',input_file=datasets['double'], name=models['double'])
# error_detection('double_match_lowercase_model_on_dobule_match/',input_file=datasets['double'], name=models['double_lower'])
# error_detection('double_match_unigram_model_on_dobule_match/',input_file=datasets['double'], name=models['double_unigram'])

# error_detection("same_blog_model_on_same_blog/", input_file=datasets['same_blog'], name=models['same_blog'])
# cross_model_analysis(double_on_double, double_lower_on_double, "double_vs_double_lower_models_on_double/")
# cross_model_analysis(double_on_double, double_unigram_on_double, "double_vs_double_unigram_models_on_double/")
#error_detection('same_shuffle_model_on_same_blog/',input_file=datasets['same_blog'], name=models['same_shuffle'])
#error_detection('same_lowercase_model_on_same_blog/',input_file=datasets['same_blog'], name=models['same_lowercase'])
error_detection('same_model_on_word_blog/',input_file=datasets['same-word'], name=models['same_blog'])
error_detection('same_model_on_lower_blog/',input_file=datasets['same-lower'], name=models['same_blog'])
error_detection('same_model_on_no_pun_blog/',input_file=datasets['same-no-pun'], name=models['same_blog'])
error_detection('same_model_on_length_blog/',input_file=datasets['double'], name=models['same_blog'])

#error_detection('same_no_pun_model_on_same_blog/',input_file=datasets['same_blog'], name=models['same_no_pun'])
#cross_model_analysis(same_on_same,'same_no_pun_model_on_same_blog',"same_vs_same_no_pun_models_on_same/")
#cross_model_analysis(same_on_same,'same_lowercase_model_on_same_blog', "same_vs_same_lowercase_models_on_same/")

# word shuffled model on normal dataset
'''
error_detection(input_file=datasets[1], name=models[0])
# normal model on shuffled dataset
error_detection(input_file=datasets[0], name=models[1])
# normal model on normal dataset
error_detection(input_file=datasets[1], name=models[1])
'''
