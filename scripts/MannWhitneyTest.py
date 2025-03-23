import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import mannwhitneyu


class MannWhitneyTest:
    """
    A class to encapsulate the Mann-Whitney U test functionality.
    """

    def run_test(self, group1, group2, alternative="two-sided"):
        """
        Runs the Mann-Whitney U test on the provided groups.
        :param alternative: 'two-sided', 'greater' (group1 > group2), or 'less' (group1 < group2).
        :return: Tuple (U statistic, p-value)
        """
        stat, p_value = mannwhitneyu(group1, group2, alternative=alternative)
        return stat, p_value

    def interpret_result(self, p_value, alpha=0.05):
        """
        Interprets the result of the test.
        :param p_value: The p-value obtained from the test.
        :param alpha: Significance level (default is 0.05).
        :return: Interpretation string.
        """
        if p_value < alpha:
            return "The result is statistically significant (reject H0)."
        else:
            return "The result is NOT statistically significant (fail to reject H0)."


# Sentiment analysis experiments results
sentiment_baseline_f1 = np.array(
    [
        0.6130105972290039,
        0.591346800327301,
        0.6055277824401856,
        0.5852092266082763,
        0.5948750138282776,
        0.610674262046814,
        0.5997408151626586,
        0.6049739599227906,
        0.5883206367492676,
        0.6033919215202331,
    ]
)

sentiment_experiment_1_f1 = np.array(
    [
        0.6178292989730835,
        0.6366260588169098,
        0.6212784051895142,
        0.6214938163757324,
        0.6103497803211212,
        0.6151657640933991,
        0.6246732354164124,
        0.6377034902572631,
        0.6339301764965057,
        0.6397077739238739,
    ]
)

sentiment_experiment_2_f1 = np.array(
    [
        0.5827255964279174,
        0.6148008942604065,
        0.5992015600204468,
        0.5914262056350708,
        0.600926685333252,
        0.6092060804367065,
        0.5887307405471802,
        0.622610068321228,
        0.6027336120605469,
        0.6052048563957214,
    ]
)

sentiment_experiment_3_f1 = np.array(
    [
        0.6131531357765198,
        0.6067179203033447,
        0.6297123670578003,
        0.6332510113716125,
        0.643090283870697,
        0.6329933166503906,
        0.6453182339668274,
        0.6330259442329407,
        0.6286608099937439,
        0.6221584916114807,
    ]
)

sentiment_experiments = np.array(
    [sentiment_experiment_1_f1, sentiment_experiment_2_f1, sentiment_experiment_3_f1]
)

test = MannWhitneyTest()

print("Sentiment analysis experiments:")
for experiment in sentiment_experiments:
    stat_one_sided, p_value_one_sided = test.run_test(
        sentiment_baseline_f1, experiment, alternative="less"
    )
    print(f"U statistic: {stat_one_sided}, p-value: {p_value_one_sided}")
    print(test.interpret_result(p_value_one_sided))
    print()


# Opinion analysis experiments results
opinion_baseline_f1 = np.array(
    [
        0.6438431143760681,
        0.6011635661125183,
        0.5680689990520478,
        0.6064783692359924,
        0.6493906021118164,
        0.6654708743095398,
        0.5854711294174194,
        0.6307744264602662,
        0.6698747396469116,
        0.5998456478118896,
    ]
)

opinion_experiment_1_f1 = np.array(
    [
        0.6535006284713745,
        0.6447460174560546,
        0.6472120404243469,
        0.6560521066188812,
        0.6628108322620392,
        0.6152590870857239,
        0.6445598483085633,
        0.6671049118041992,
        0.6406868815422058,
        0.6571347832679748,
    ]
)

opinion_experiment_2_f1 = np.array(
    [
        0.4916576504707336,
        0.5872839450836181,
        0.6397462129592896,
        0.5616132974624634,
        0.654293954372406,
        0.6623787999153137,
        0.6312576293945312,
        0.5589958786964416,
        0.6178455233573914,
        0.6413082003593444,
    ]
)

opinion_experiment_3_f1 = np.array(
    [
        0.6546963691711426,
        0.6535470128059387,
        0.6108193635940552,
        0.6607183456420899,
        0.6632133245468139,
        0.6294217467308044,
        0.6362064719200134,
        0.657424533367157,
        0.6752328515052796,
        0.647264564037323,
    ]
)

opinion_experiments = np.array(
    [opinion_experiment_1_f1, opinion_experiment_2_f1, opinion_experiment_3_f1]
)

test = MannWhitneyTest()

print("Opinion analysis experiments:")
for experiment in opinion_experiments:
    stat_one_sided, p_value_one_sided = test.run_test(
        opinion_baseline_f1, experiment, alternative="less"
    )
    print(f"U statistic: {stat_one_sided}, p-value: {p_value_one_sided}")
    print(test.interpret_result(p_value_one_sided))
    print()

# testing equal p-values

def compute_statistics(data, label):
    mean = np.mean(data)
    median = np.median(data)
    print(f"{label} mean: {mean:.6f}, median: {median:.6f}")
    return mean, median


print("\nSentiment Analysis:")
mean_sentiment_baseline, median_sentiment_baseline = compute_statistics(
    sentiment_baseline_f1, "Baseline"
)
mean_sentiment_exp1, median_sentiment_exp1 = compute_statistics(
    sentiment_experiment_1_f1, "Experiment 1"
)
mean_sentiment_exp3, median_sentiment_exp3 = compute_statistics(
    sentiment_experiment_3_f1, "Experiment 3"
)

print("\nOpinion Detection:")
mean_opinion_baseline, median_opinion_baseline = compute_statistics(
    opinion_baseline_f1, "Baseline"
)
mean_opinion_exp1, median_opinion_exp1 = compute_statistics(
    opinion_experiment_1_f1, "Experiment 1"
)
mean_opinion_exp3, median_opinion_exp3 = compute_statistics(
    opinion_experiment_3_f1, "Experiment 3"
)

# boxplots

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# sentiment analysis
axes[0].boxplot(
    [sentiment_baseline_f1, sentiment_experiment_1_f1, sentiment_experiment_3_f1],
    labels=["Baseline", "RQ1", "RQ3"],
)
axes[0].set_title("Rozdelenie F1-skóre pre sentimentovú klasifikáciu")
axes[0].set_ylabel("F1-skóre")
axes[0].set_xlabel("Modely")
axes[0].grid(True)

# opinion detection
axes[1].boxplot(
    [opinion_baseline_f1, opinion_experiment_1_f1, opinion_experiment_3_f1],
    labels=["Baseline", "Experiment 1", "Experiment 3"],
)
axes[1].set_title("Rozdelenie F1-skóre pre detekciu názorov")
axes[1].set_ylabel("F1-skóre")
axes[1].set_xlabel("Modely")
axes[1].grid(True)

plt.tight_layout()
plt.show()
