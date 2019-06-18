import pandas as pd

df1 = pd.read_csv('data_image/node020/lyapnov_end_err.dat',
                header=None,
                names=('G', 'NU', 'error'),
                delim_whitespace=True)

df2 = pd.read_csv('data_image/node020/lyapnov_end_ly.dat',
                header=None,
                names=('G', 'NU', 'lyapnov'),
                delim_whitespace=True)
                
df1 = df1.set_index(['G', 'NU'], drop=True)
df2 = df2.set_index(['G', 'NU'], drop=True)

print(df1)
print(df1['error'].min())
print(df1['error'].idxmin())
errmin=df1['error'].idxmin()


print(df2.loc[errmin] )
