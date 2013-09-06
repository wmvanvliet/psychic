import golem, psychic
from matplotlib import pyplot as plt
from annotate import annotate_horiz

trials = golem.DataSet.load('../../data/priming-trials.dat')

f = plt.figure(figsize=(12,4))
f.add_axes([0.05, 0.2, 0.38, 0.7])
trials2 = psychic.nodes.Baseline([.2, 1]).train_apply(trials, trials)
psychic.plot_erp(trials2.lix[['P3'],:,:], fig=f, pval=0, vspace=15)
annotate_horiz(-.19, -0.01, 5, 'baseline period', 0.5) 
plt.gca().legend().set_visible(False)
plt.title('Uncorrected')

f.add_axes([0.55, 0.2, 0.38, 0.7])
trials2 = psychic.nodes.Baseline([-0.2, 0]).train_apply(trials, trials)
psychic.plot_erp(trials2.lix[['P3'],:,:], fig=f, pval=0, vspace=15)
annotate_horiz(-.19, -0.01, 5, 'baseline period', 0.5) 
#plt.legend(loc='lower right')
plt.gca().legend().set_visible(False)
plt.title('Baseline corrected')

plt.savefig('baseline.png')
