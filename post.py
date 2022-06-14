import torch
import optuna

study = torch.load('study.pth')

df = study.trials_dataframe()
print(df)

#FLAGS.config.model.num_scales = trial.suggest_int("num_scales", 900, 1200, step=100)
#FLAGS.config.model.beta_max = trial.suggest_discrete_uniform("beta_max", 10, 30, 2)
#FLAGS.config.model.nonlinearity = trial.suggest_categorical("nonlinearity", ["swish", "relu"])
#FLAGS.config.optim.lr = trial.suggest_float("lr", 1e-4, 4e-4, step=1e-4)
#FLAGS.config.model.discount_sigma = trial.suggest_float("discount_sigma", 0.7, 1.2, step=0.1)

'''
fig = optuna.visualization.matplotlib.plot_contour(study, params=['num_scales', 'beta_max'])
fig.figure.savefig('contour_n_b.png')

fig = optuna.visualization.matplotlib.plot_contour(study, params=['num_scales', 'lr'])
fig.figure.savefig('contour_n_l.png')

fig = optuna.visualization.matplotlib.plot_contour(study, params=['num_scales', 'discount_sigma'])
fig.figure.savefig('contour_n_d.png')

fig = optuna.visualization.matplotlib.plot_contour(study, params=['discount_sigma', 'lr'])
fig.figure.savefig('contour_d_l.png')

fig = optuna.visualization.matplotlib.plot_contour(study, params=['lr', 'nonlinearity'])
fig.figure.savefig('contour_l_non.png')
'''

fig = optuna.visualization.plot_param_importances(study)
fig.write_image('importancce.png')


