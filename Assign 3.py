import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import errors as err
import sklearn.metrics as skmet
from sklearn import cluster

def get_data_frames(filename,countries,indicator):
    '''

    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    # Get datafarme information.
    df.info()
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by countries
    df = df.loc[df['Country Name'].isin(countries)]
    # To filter data by indicator code.
    df = df.loc[df['Indicator Code'].eq(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    # Deleting country code column.
    del df2['Country Code']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Indicator Name','Indicator Code']
                          ,'Country Name').reset_index()
    
    df_countries = df
    df_years = df2
    
    # Cleaning data droping nan values.
    df_countries.dropna()
    df_years.dropna()
    
    return df_countries, df_years

def get_data_frames1(filename,indicator):
    '''
    Parameters
    ----------
    filename : Text
        Name of the file to read data.
    countries : List
        List of countries to filter the data.
    indicator : Text
        Indicator Code to filter the data.

    Returns
    -------
    df_countries : DATAFRAME
        This dataframe contains countries in rows and years as column.
    df_years : DATAFRAME
        This dataframe contains years in rows and countries as column..

    '''
    # Read data using pandas in a dataframe.
    df = pd.read_csv(filename, skiprows=(4), index_col=False)
    
    # To clean data we need to remove unnamed column.
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # To filter data by indicator codes.
    df = df.loc[df['Indicator Code'].isin(indicator)]
    
    # Using melt function to convert all the years column into rows as 1 column
    df2 = df.melt(id_vars=['Country Name','Country Code','Indicator Name'
                           ,'Indicator Code'], var_name='Years')
    
    # Deleting country code column.
    del df2['Indicator Name']
    # Using pivot table function to convert countries from rows to separate 
    # column for each country.   
    df2 = df2.pivot_table('value',['Years','Country Name','Country Code']
                          ,'Indicator Code').reset_index()
    
    df_countries = df
    df_indticators = df2
    
    # Droping nun values from given data.
    df_countries.dropna()
    df_indticators.dropna()
    
    return df_countries, df_indticators


def poly(x, a, b, c, d):
    '''
    Cubic polynominal for the fitting
    '''
    y = a*x**3 + b*x**2 + c*x + d
    return y

def exp_growth(t, scale, growth):
    ''' 
    Computes exponential function with scale and growth as free parameters
    '''
    f = scale * np.exp(growth * (t-1960))
    return f

def logistics(t, scale, growth, t0):
    ''' 
    Computes logistics function with scale, growth raat
    and time of the turning point as free parameters
    '''
    f = scale / (1.0 + np.exp(-growth * (t - t0)))
    return f

def norm(array):
    '''
    Returns array normalised to [0,1]. Array can be a numpy array
    or a column of a dataframe
    '''
    min_val = np.min(array)
    max_val = np.max(array)
    scaled = (array-min_val) / (max_val-min_val)
    return scaled

def norm_df(df, first=0, last=None):
    '''
    Returns all columns of the dataframe normalised to [0,1] with the
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds
    '''
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
    return df


def map_corr(df, size=12):
    """Function creates heatmap of correlation matrix for each pair
    of columns
    ↪→in the dataframe.
    Input:
    df: pandas DataFrame
    size: vertical and horizontal size of the plot (in inch)
    """
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr, cmap='RdBu')
    # setting ticks to column names
    plt.xticks(range(len(corr.columns)),
               ['Electric power consumption (kWh per capita)',
                'Electricity production from natural gas sources (% of total)',
                'Electricity production from oil sources (% of total)',
                'Total Population'],
               rotation=90)
    plt.yticks(range(len(corr.columns)),
               ['Electric power consumption (kWh per capita)',
                'Electricity production from natural gas sources (% of total)',
                'Electricity production from oil sources (% of total)',
                'Total Population'])


#==============================================================================
# Data fitting for Australia Population with prediction
#==============================================================================

countries = ['India','Australia','Canada','United States','China',
             'United Kingdom']
# calling functions to get dataframes and use for plotting graphs.
df_c, df_y = get_data_frames('C:/Users/User/Desktop/Assign 3.csv',countries,
                             'SP.POP.TOTL')

df_y['Years'] = df_y['Years'].astype(int)

popt, covar = curve_fit(exp_growth, df_y['Years'], df_y['Australia'])
print("Fit parameter", popt)
# use *popt to pass on the fit parameters
df_y['Australia_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y["Australia"], label='data')
plt.plot(df_y['Years'], df_y['Australia_exp'], label='fit')
plt.legend()
plt.title("First fit attempt")
plt.xlabel("Year")
plt.ylabel("Australia Population")
plt.show()

# find a feasible start value the pedestrian way
# the scale factor is way too small. The exponential factor too large.
# Try scaling with the 1950 population and a smaller exponential factor
# decrease or increase exponential factor until rough agreement is reached
# growth of 0.07 gives a reasonable start value
popt = [7e8, 0.01]
df_y['Australia_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['Australia'], label='data')
plt.plot(df_y['Years'], df_y['Australia_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("Australia Population")
plt.title("Improved start value")
plt.show()

# fit exponential growth
popt, covar = curve_fit(exp_growth, df_y['Years'],df_y['Australia'], p0=[7e8, 0.02])
# much better
print("Fit parameter", popt)
df_y['Australia_exp'] = exp_growth(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['Australia'], label='data')
plt.plot(df_y['Years'], df_y['Australia_exp'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("Australia Population")
plt.title("Final fit exponential growth")
plt.show()


# estimated turning year: 1990
# population in 1990: about 1135185000
# kept growth value from before
# increase scale factor and growth rate until rough fit
popt = [17065100, 0.02, 1990]
df_y['Australia_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['Australia'], label='data')
plt.plot(df_y['Years'], df_y['Australia_log'], label='fit')
plt.legend()
plt.xlabel("Year")
plt.ylabel("Australia Population")
plt.title("Improved start value")
plt.show()

popt, covar = curve_fit(logistics,  df_y['Years'],df_y['Australia'],
                        p0=[17065100, 0.02, 1990])
print("Fit parameter", popt)
df_y['Australia_log'] = logistics(df_y['Years'], *popt)
plt.figure()
plt.plot(df_y['Years'], df_y['Australia'], label='Data Value')
plt.plot(df_y['Years'], df_y['Australia_log'], label='Fitting Value')
plt.legend()
plt.xlabel("Year")
plt.ylabel("Australia Population")
plt.title("Logistic Function", color="blue", fontsize=25)


# extract the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df_y['Years'], logistics, popt, sigma)
plt.figure()
plt.plot(df_y['Years'], df_y['Australia'], label='Given Data')
plt.plot(df_y['Years'], df_y['Australia_log'], label='fitting value',
         color='black')
plt.fill_between(df_y['Years'], low, up, alpha=0.4, label='error range')
plt.legend()
plt.title("Logistics function", color = "red", fontsize=32)
plt.xlabel("Year", fontsize=20, color = "violet")
plt.ylabel("Australia Population", fontsize=17)

# Your code for plotting the graph

# Add text below the graph
text = ("- Here the given graph represents Australian population with logistic"
        " value\n and the highlight represents the error value for the data")
plt.text(0.5, -0.31, text, ha='center', va='center',
         transform=plt.gca().transAxes)

plt.show()


print("Forcasted population")
low, up = err.err_ranges(2040, logistics, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err.err_ranges(2070, logistics, popt, sigma)
print("2070 between ", low, "and", up)
low, up = err.err_ranges(2100, logistics, popt, sigma)
print("2100 between ", low, "and", up)

print("Forcasted population")
low, up = err.err_ranges(2040, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)
low, up = err.err_ranges(2070, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2070:", mean, "+/-", pm)
low, up = err.err_ranges(2100, logistics, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2100:", mean, "+/-", pm)


#==============================================================================
# Clustering Analysis (k-means Clustering)
#==============================================================================

indicators = ['EG.USE.ELEC.KH.PC','EG.ELC.NGAS.ZS','EG.ELC.PETR.ZS',
              'SP.POP.TOTL']
df_y, df_i = get_data_frames1('C:/Users/User/Desktop/Assign 3.csv'
                             ,indicators)


df_i = df_i.loc[df_i['Years'].eq('2015')]
df_i = df_i.loc[~df_i['Country Code'].isin(["USA", "AUS"])]

df_i.dropna()

# Heat Map Plot
map_corr(df_i)
plt.show()

# Scatter Matrix Plot
pd.plotting.scatter_matrix(df_i, figsize=(9.0, 9.0))
plt.suptitle("Scatter Matrix Plot For All Countries", fontsize=20)
 # helps to avoid overlap of labels
plt.show()


# extract columns for fitting
df_fit = df_i[['EG.ELC.NGAS.ZS','EG.ELC.PETR.ZS']].copy()
# normalise dataframe and inspect result
# normalisation is done only on the extract columns. .copy() prevents
# changes in df_fit to affect df_fish. This make the plots with the
# original measurements
df_fit = norm_df(df_fit)
df_fit = df_fit.dropna()
print(df_fit.describe())



for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(df_fit)
    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))


# Plot cluster in scateer 
kmeans = cluster.KMeans(n_clusters=3)
kmeans.fit(df_fit)

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(8.0, 6.0))

# Individual colours can be assigned to symbols. The label l is used to the
# select the l-th number from the colour table.
scatter = plt.scatter(df_fit['EG.ELC.NGAS.ZS'], df_fit['EG.ELC.PETR.ZS'],
                      c=labels, cmap="Accent")

# Create a legend for the scatter plot
legend_elements = scatter.legend_elements()
labels = ['Data Value {}'.format(i) for i in range(len(legend_elements[0]))]
plt.legend(legend_elements[0], labels, title='Values', prop={"size": 14})

# Show cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)

plt.xlabel("Electricity production from\n natural gas sources (% of total)",
           fontsize=18, color="green")
plt.ylabel("Electricity production from\n oil sources (% of total)",
           fontsize=18, color="blue")
plt.title("Clusters For All Countries", color="m", fontsize=45)
plt.show()



def clean(x):

    # count the number of missing values in each column of the DataFrame
    x.isnull().sum()
    # fill any missing values with 0 and update the DataFrame in place
    x.fillna(0, inplace=True)

    return

clean(df_fit)

df_fit_trans = df_fit.transpose()
print(df_fit_trans.head())


