import numpy
from pandas import DataFrame, read_csv
import matplotlib.pyplot as plt


dfr = read_csv('bench_plot_onnxruntime_linreg.csv')
dfr["speedup"] = dfr["time_skl"] / dfr["time_ort"]
dfc = read_csv('bench_plot_onnxruntime_logreg.csv')
dfc = dfc[(dfc.method == "predict_proba") & (dfc.fit_intercept == True)]
dfc["speedup"] = dfc["time_skl"] / dfc["time_ort"]

nfeats = [10, 100]
fig, axs = plt.subplots(1, len(nfeats) * 2, figsize=(14, 4), sharey=True)


def autolabel(ax, rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate('%1.1fx' % height,
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom',
                    fontsize=8)

names = ["LinearRegression", "LogisticRegression"]
pos = 0
for name, df in zip(names, [dfr, dfc]):
    for nf in nfeats:
        ax = axs[pos]
        sub = df[df.nfeat == nf]
        labels = sub.n_obs
        means = sub.speedup

        x = numpy.arange(len(labels))
        width = 0.90

        rects1 = ax.bar(x, means, width, label='Speedup')

        if pos == 0:
            ax.set_ylabel('Speedup')
        ax.set_title('%s %d features' % (name, nf))
        ax.set_xlabel('batch size')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        autolabel(ax, rects1)
        for tick in ax.xaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        for tick in ax.yaxis.get_major_ticks():
            tick.label.set_fontsize(8) 
        pos += 1

fig.tight_layout()
fig.savefig("linear_model.png")
plt.show()
