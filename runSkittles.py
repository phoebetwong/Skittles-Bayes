from Skittles import *
#Flat prior
Model1 = np.linspace(0, 1, 100)
prior1 = [0.01]*100
#normal distribution 
Model2 = np.linspace(0, 1, 100)
prior2 = stat.norm.pdf(range(-50,50),0.5, 2)
Model3 = np.linspace(0, 1, 1000)
prior3 = stat.norm.pdf(range(-500,500),0.5, 2)

#Update prob using cumulative data
runLoop(Model1, prior1, 10, 1)

#Update prob using last 1 draws
runLoop_2(Model1, prior1, 10, 1)

#Update prob using last 5 draws
runLoop_2(Model1, prior1, 10, 5)

#boo