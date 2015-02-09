Model1 = np.linspace(0, 1, 100)
prior1 = [0.01]*100
Model2 = np.linspace(0, 1, 100)
prior2 = stat.norm.pdf(range(-50,50),0.5, 2)
Model3 = np.linspace(0, 1, 1000)
prior3 = stat.norm.pdf(range(-500,500),0.5, 2)
plt.plot(Model, prior)
