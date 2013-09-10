import psychic
import golem
from matplotlib import pyplot as plt

trials = golem.DataSet.load(psychic.find_data_path('priming-trials.dat'))
trials = psychic.nodes.Baseline((-0.2, 0)).train_apply(trials, trials)
psychic.plot_erp(trials.lix[['Fz', 'Cz', 'Pz'], :, :], fwer=None)
plt.savefig('plot_erp.png')

erp = psychic.nodes.ERP().apply(trials)
psychic.plot_erp(erp.lix[['Fz', 'Cz', 'Pz'], :, :])
plt.savefig('plot_erp_erp.png')
