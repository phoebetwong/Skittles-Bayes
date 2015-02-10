import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stat

# prior = stat.norm.pdf(range(-50,50),0.9, 2)
# Model = np.linspace(0, 1, 100)


def runLoop(Theta = np.array([0.1, 0.2]),  pTheta = np.array([0.5, 0.5]), nLoop=10, plotSpace = 2):
    """
    Plots prior, likelihood and posterior probability over nLoop draws. 
    Each draw does NOT updates the prior probability.  Old data is kept and
    the posterior and likelihood are computed from the initial priors and 
    cumulative data

    :Theta: array of possible outcomes
    :pTheta: array of prior probabilities of outcomes
    :nLoop: number of trials to run
    :plotSpace: number of trials between plot updates
    """
    # Dr. Lynch's two models 
    # 10% Green skittles
    # 20% Green skittles
    # Prior belief:
    # Equal belief in either 
    # Prior belief 2:
    # I believe there is only a 10% chance the bag was altered
    # pTheta = [0.1, 0.9]
    
    pTheta = pTheta/np.sum(pTheta) 
    
    # Initial belief
    pTheta0 = pTheta
    pTheta = np.array(pTheta)
    print (pTheta)
    
    data = []

    for i in range(0, nLoop):
        bagV = np.random.uniform(0,1)

        if bagV >= 0.9: 
            data.append(1)
        else:
            data.append(0)
        plotSpace = int(plotSpace)
        nGreens = np.sum( data )
        nOther = len( data ) - nGreens
        if i % plotSpace == 0:
            print "Cumulative number of green Skittles: ", nGreens
            # I've left out the factor of N choose nGreen that is common 
            # to pData and pDataGivenTheta
                        
            # Compute the likelihood of the data for each value of theta:
            pDataGivenTheta = Theta**nGreens * (1-Theta)**nOther
            # Compute the posterior:
            pData = sum( pDataGivenTheta * pTheta )
            # Use Bayes' rule!
            pThetaGivenData = pDataGivenTheta * pTheta / pData   
            checkNorm = sum(pThetaGivenData)
            hBins = np.linspace(0,1, 101)
            fig = plt.figure()
            ax1 = fig.add_subplot(4,1,1)
            ax1.hist(Theta, weights = pTheta0, bins = hBins, alpha = 0.4, label = "Starting Prior")
            ax1.legend()
            plt.xlim(0, 1)
            
            ax2 = fig.add_subplot(4,1,2)
            ax2.hist(Theta, weights = pTheta, bins = hBins, alpha = 0.4, label = "Current Prior")
            ax2.legend()
            plt.xlim(0, 1)
            ax3 = fig.add_subplot(4,1,3)
            ax3.hist(Theta, weights = pDataGivenTheta, bins = hBins, alpha = 0.4, label = "Likelihood")
            ax3.legend()
            plt.xlim(0, 1)
            ax4 = fig.add_subplot(4,1,4)
            ax4.hist(Theta, weights = pThetaGivenData, bins = hBins, alpha = 0.4, label = "Posterior")
            ax4.legend()
            plt.xlim(0, 1)
            plt.show()
            #pTheta = pThetaGivenData

    print "Cumulative number of green Skittles: ", nGreens
    # Compute the likelihood of the data for each value of theta:
    pDataGivenTheta = Theta**nGreens * (1-Theta)**nOther
    # Compute the posterior:
    pData = sum( pDataGivenTheta * pTheta )
    # Use Bayes' rule!
    pThetaGivenData = pDataGivenTheta * pTheta / pData   
    checkNorm = sum(pThetaGivenData)
    hBins = np.linspace(0,1, 101)
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.hist(Theta, weights = pTheta0, bins = hBins, alpha = 0.4, label = "Prior")
    ax1.legend()
    plt.xlim(0, 1)
    ax2 = fig.add_subplot(4,1,2)
    ax2.hist(Theta, weights = pTheta, bins = hBins, alpha = 0.4, label = "Current Prior")
    ax2.legend()
    plt.xlim(0, 1)
    ax3 = fig.add_subplot(4,1,3)
    ax3.hist(Theta, weights = pDataGivenTheta, bins = hBins, alpha = 0.4, label = "Likelihood")
    ax3.legend()
    plt.xlim(0, 1)
    ax4 = fig.add_subplot(4,1,4)
    ax4.hist(Theta, weights = pThetaGivenData, bins = hBins, alpha = 0.4, label = "Posterior")
    ax4.legend()
    plt.xlim(0, 1)
    plt.show()


def runLoop_2(Theta = np.array([0.1, 0.2]),  pTheta = np.array([0.5, 0.5]), nLoop=10, plotSpace = 1):
    """
    Plots prior, likelihood and posterior probability over nLoop draws. 
    Each draw updates the prior probability i.e. prior = posterior and 
    clears the 'old' data
    
    :Theta: array of possible outcomes
    :pTheta: array of prior probabilities of outcomes
    :nLoop: number of trials to run
    :plotSpace: number of trials between plot updates
    """
    # Dr. Lynch's two models 
    # 10% Green skittles
    # 20% Green skittles
    # Prior belief:
    # Equal belief in either 
    # Prior belief 2:
    # I believe there is only a 10% chance the bag was altered
    # pTheta = [0.1, 0.9]
    
    pTheta = pTheta/np.sum(pTheta) 
    
    # Initial belief
    pTheta0 = pTheta
    pTheta = np.array(pTheta)
    print (pTheta)
    
    data = []
    cumulativeG = 0
    for i in range(0, nLoop):
        bagV = np.random.uniform(0,1)

        if bagV >= 0.9: 
            data.append(1)
        else:
            data.append(0)
        if i % plotSpace == 0:
            plotSpace = int(plotSpace)
            nGreens = np.sum( data )
            nOther = len( data ) - nGreens
            cumulativeG += nGreens
            print "Number of green Skittles in this bag of ", plotSpace ,  " = ", nGreens
            # I've left out the factor of N choose nGreen that is common 
            # to pData and pDataGivenTheta                        
            # Compute the likelihood of the data for each value of theta:
            pDataGivenTheta = Theta**nGreens * (1-Theta)**nOther
            # Compute the posterior:
            pData = sum( pDataGivenTheta * pTheta )
            # Use Bayes' rule!
            pThetaGivenData = pDataGivenTheta * pTheta / pData   
            checkNorm = sum(pThetaGivenData)
            hBins = np.linspace(0,1, 101)
            fig = plt.figure()
            ax1 = fig.add_subplot(4,1,1)
            ax1.hist(Theta, weights = pTheta0, bins = hBins, alpha = 0.4, label = "Starting Prior")
            ax1.legend()
            plt.xlim(0, 1)
            ax2 = fig.add_subplot(4,1,2)
            ax2.hist(Theta, weights = pTheta, bins = hBins, alpha = 0.4, label = "Current Prior")
            ax2.legend()
            plt.xlim(0, 1)
            ax3 = fig.add_subplot(4,1,3)
            ax3.hist(Theta, weights = pDataGivenTheta, bins = hBins, alpha = 0.4, label = "Likelihood")
            ax3.legend()
            plt.xlim(0, 1)
            ax4 = fig.add_subplot(4,1,4)
            ax4.hist(Theta, weights = pThetaGivenData, bins = hBins, alpha = 0.4, label = "Posterior")
            ax4.legend()
            plt.xlim(0, 1)
            plt.show()
            pTheta = pThetaGivenData
            data = []
    print "Number of green Skittles in this bag: ", nGreens
    print "Total number of green skittles found: ", cumulativeG
    # Compute the likelihood of the data for each value of theta:
    pDataGivenTheta = Theta**nGreens * (1-Theta)**nOther
    # Compute the posterior:
    pData = sum( pDataGivenTheta * pTheta )
    # Use Bayes' rule!
    pThetaGivenData = pDataGivenTheta * pTheta / pData   
    checkNorm = sum(pThetaGivenData)
    hBins = np.linspace(0,1, 100)
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax1.hist(Theta, weights = pTheta0, bins = hBins, alpha = 0.4, label = "Prior")
    ax1.legend()
    plt.xlim(0, 1)
    ax2 = fig.add_subplot(4,1,2)
    ax2.hist(Theta, weights = pTheta, bins = hBins, alpha = 0.4, label = "Current Prior")
    ax2.legend()
    plt.xlim(0, 1)
    ax3 = fig.add_subplot(4,1,3)
    ax3.hist(Theta, weights = pDataGivenTheta, bins = hBins, alpha = 0.4, label = "Likelihood")
    ax3.legend()
    plt.xlim(0, 1)
    ax4 = fig.add_subplot(4,1,4)
    ax4.hist(Theta, weights = pThetaGivenData, bins = hBins, alpha = 0.4, label = "Posterior")
    ax4.legend()
    plt.xlim(0, 1)
    plt.show()
    pTheta = pThetaGivenData

def SingleShot(N):
    # Dr. Lynch's two models 
    # 10% Green skittles
    # 20% Green skittles
    Theta = np.array([0.1, 0.2])
    
    # Prior belief:
    # Equal belief in either 
    pTheta = np.array([0.5, 0.5])
    
    # Prior belief 2:
    # I believe there is only a 10% chance the bag was altered
    # pTheta = [0.1, 0.9]
    
    pTheta = pTheta/np.sum(pTheta) 
    
    # Initial belief
    pTheta0 = pTheta
    pTheta = np.array(pTheta)
    print "Prior: ", (pTheta)
    print "Models: ", (Theta)

    nGreens = round(N*1.0/10.0)
    nOther = N - nGreens
    pDataGivenTheta = Theta**nGreens * (1-Theta)**nOther
    # Compute the posterior:
    pData = sum( pDataGivenTheta * pTheta )
    # Use Bayes' rule!
    pThetaGivenData = pDataGivenTheta * pTheta / pData   
    checkNorm = sum(pThetaGivenData)
    pTheta = pThetaGivenData
    
