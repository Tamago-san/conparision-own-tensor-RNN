import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

lag=100


df = pd.read_csv('./data/output_Runge_Lorenz.csv',
#                    usecols=[1],
                    engine='python',
                    header =None
                )
print(len(df))
lorenz_acf= pd.DataFrame(sm.tsa.stattools.acf(df[0], nlags=lag),
                  columns=['X_acf'])
lorenz_acf['Y_acf']= pd.DataFrame(sm.tsa.stattools.acf(df[1], nlags=lag))
lorenz_acf['Z_acf']= pd.DataFrame(sm.tsa.stattools.acf(df[2], nlags=lag))

#lorenz_acf = sm.tsa.stattools.acf(df[0], nlags=40)
#lorenz_acf = sm.tsa.stattools.acf(df[0], nlags=40)

print(lorenz_acf)

#plt.figure()
lorenz_acf.plot()
plt.show()