import numpy as np
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
        0.6220856189727784,
        0.6250235795974731,
        0.6180454730987549,
        0.5985336065292358,
        0.6282204985618591,
        0.6275418162345886,
        0.6137187480926514,
        0.6085644125938415,
        0.6073259592056275,
        0.6260121464729309,
    ]
)

sentiment_experiment_1_f1 = np.array(
    [
        0.6310455799102783,
        0.631223839521408,
        0.6316366791725159,
        0.6189098477363586,
        0.6486050903797149,
        0.6275447249412537,
        0.632792055606842,
        0.6124779462814331,
        0.614151555299759,
        0.6201420307159424,
    ]
)

sentiment_experiment_2_f1 = np.array(
    [
        0.5952069520950317,
        0.6212198853492736,
        0.6062157154083252,
        0.6149736166000366,
        0.597359037399292,
        0.5722776412963867,
        0.6029300451278686,
        0.6360800385475158,
        0.5858291864395142,
        0.6039724588394165,
    ]
)

sentiment_experiment_3_f1 = np.array(
    [
        0.6148154497146606,
        0.6456211566925049,
        0.6169899463653564,
        0.6067038893699646,
        0.6239444851875305,
        0.6247076749801636,
        0.6230051398277283,
        0.6289791584014892,
        0.6450015664100647,
        0.6282133102416992,
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
        0.6298790097236633,
        0.6620452523231506,
        0.6302444100379944,
        0.6107773780822754,
        0.6215250015258789,
        0.6343305706977844,
        0.642332661151886,
        0.6440966963768006,
        0.6405023097991943,
        0.6438658118247986,
    ]
)

opinion_experiment_3_f1 = np.array(
    [
        0.5969224870204926,
        0.6487890243530273,
        0.6321978747844696,
        0.6193499684333801,
        0.6052219659090042,
        0.6495251536369324,
        0.6183151602745056,
        0.6688963651657105,
        0.6775144577026367,
        0.6244686126708985,
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
