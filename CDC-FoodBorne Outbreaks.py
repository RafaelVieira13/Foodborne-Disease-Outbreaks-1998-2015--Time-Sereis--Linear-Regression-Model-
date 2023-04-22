#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import statistics as sts
from scipy import stats
from datetime import datetime
plt.style.use("ggplot")


# In[2]:


outbreaks=pd.read_csv("outbreaks.csv")


# In[3]:


outbreaks


# In[4]:


outbreaks.info()


# # Deal with Datetime Data

# __Create a Columns "Date" with values from column "Year" and "Month" and change the data typo to datetime__

# In[5]:


outbreaks["Month"].value_counts()


# In[6]:


outbreaks["Month"]=outbreaks["Month"].str.replace("January","01")


# In[7]:


outbreaks["Month"]=outbreaks["Month"].str.replace("February","02")


# In[8]:


outbreaks["Month"]=outbreaks["Month"].str.replace("March","03")


# In[9]:


outbreaks["Month"]=outbreaks["Month"].str.replace("April","04")


# In[10]:


outbreaks["Month"]=outbreaks["Month"].str.replace("May","05")


# In[11]:


outbreaks["Month"]=outbreaks["Month"].str.replace("June","06")


# In[12]:


outbreaks["Month"]=outbreaks["Month"].str.replace("July","07")


# In[13]:


outbreaks["Month"]=outbreaks["Month"].str.replace("August","08")


# In[14]:


outbreaks["Month"]=outbreaks["Month"].str.replace("September","09")


# In[15]:


outbreaks["Month"]=outbreaks["Month"].str.replace("October","10")


# In[16]:


outbreaks["Month"]=outbreaks["Month"].str.replace("November","11")


# In[17]:


outbreaks["Month"]=outbreaks["Month"].str.replace("December","12")


# In[18]:


outbreaks["Month"].value_counts()


# In[19]:


outbreaks.info()


# In[20]:


outbreaks=  outbreaks.astype({'Year':'str'})


# In[21]:


outbreaks=  outbreaks.astype({'Month':'str'})


# In[22]:


outbreaks.info()


# In[23]:


outbreaks["Date"]= outbreaks["Year"] + "-" + outbreaks["Month"]


# In[24]:


outbreaks.head()


# In[25]:


outbreaks.drop(columns=["Year","Month"],inplace=True)


# In[26]:


outbreaks=outbreaks[["Date","State","Location","Food","Ingredient","Species","Serotype/Genotype","Status","Illnesses","Hospitalizations","Fatalities"]]


# In[27]:


outbreaks


# In[28]:


outbreaks.to_csv("outbreaks2.csv")


# __Change the "Date" Column type from string to datetime: Year-Month__

# In[29]:


dateparse=lambda dates:datetime.strptime(dates,"%Y-%m")
outbreaks=pd.read_csv("outbreaks2.csv",parse_dates=["Date"],date_parser=dateparse)


# In[30]:


outbreaks


# In[31]:


outbreaks.drop(columns=["Unnamed: 0"],inplace=True)


# In[32]:


outbreaks.head()


# In[33]:


outbreaks.describe()


# In[34]:


outbreaks.info()


# # Exploratory Analysis

# 1. Categorical Data

# In[35]:


outbreaks["Date"].value_counts()


# In[36]:


outbreaks["State"].value_counts()


# In[37]:


outbreaks["Food"].value_counts()


# In[38]:


outbreaks["Ingredient"].value_counts()


# In[39]:


outbreaks["Species"].value_counts()


# In[40]:


outbreaks["Serotype/Genotype"].value_counts()


# In[41]:


outbreaks["Species"].value_counts()


# 2. Numerical Data

# _Illnesses_

# In[42]:


outbreaks["Illnesses"].describe()


# In[43]:


outbreaks["Hospitalizations"].describe()


# In[44]:


outbreaks["Fatalities"].describe()


# # Taking Care of Missing Values

# In[45]:


outbreaks.isnull().sum()


# In[46]:


outbreaks["Hospitalizations"].fillna(0,inplace=True)


# In[47]:


outbreaks["Fatalities"].fillna(0,inplace=True)


# In[48]:


outbreaks.isnull().sum()


# # First Goal: Time Series
# >Are foodborne disease outbreaks increasing or decreasing?
# 
# >What was the year with the most illness, hospitalizations and deaths?

# Time Serie case!!

# In[49]:


from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
from statsmodels.tsa.seasonal import seasonal_decompose
get_ipython().system('pip install pmdarima')
from pmdarima.arima import auto_arima


# In[50]:


outbreaks


# __Define Column "Date" as index__

# In[51]:


st_outbreaks=outbreaks.set_index("Date")


# In[52]:


st_outbreaks.head()


# __Analysing the Data per Year__

# In[53]:


outbreaks_per_year=outbreaks.groupby(pd.Grouper(key="Date",freq="Y")).agg(Nº_of_outbreaks=("Date","count"),
                              Nº_of_Illnesses=("Illnesses","sum"),
                              Nº_of_Hospitalizations=("Hospitalizations","sum"),
                              Nº_of_Fatalities=("Fatalities","sum"))
outbreaks_per_year


# In[54]:


fig=plt.figure(figsize=(15,15))
plt.style.use("ggplot")

fig.add_subplot(4,2,1)
outbreaks_per_year["No_of_outbreaks"].plot()
plt.title("Nº of Outbreaks per Year")
plt.xlabel("Year")
plt.ylabel("Nº of Outbreaks")

fig.add_subplot(4,2,2)
outbreaks_per_year["No_of_Illnesses"].plot()
plt.title("Nº of Illness per Year")
plt.xlabel("Year")
plt.ylabel("Nº of Illness")

fig.add_subplot(4,2,3) 
outbreaks_per_year["No_of_Hospitalizations"].plot()
plt.title("Nº of Hospitalizations per Year")
plt.xlabel("Year")
plt.ylabel("Nº of Hospitalizations")

fig.add_subplot(4,2,4)
outbreaks_per_year["No_of_Fatalities"].plot()
plt.title("Nº of Fatalities per Year")
plt.xlabel("Year")
plt.ylabel("Nº of Fatalities")

plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.show()


# __Analyzing the Data per Month__

# In[55]:


# Create the Time Serie
st_outbreaks=outbreaks.set_index("Date")
st_outbreaks


# In[56]:


outbreaks


# In[57]:


outb2=outbreaks


# In[58]:


outb2['Month'] = outb2['Date'].dt.strftime('%m')


# In[59]:


outb2


# In[60]:


outbreaks_per_month2=outb2.groupby("Month").agg(Nº_of_outbreaks=("Month","count"))
outbreaks_per_month2


# In[61]:


outbreaks_per_month=st_outbreaks.groupby([lambda x: x.month]).sum()
outbreaks_per_month


# In[62]:


fig=plt.figure(figsize=(15,15))

fig.add_subplot(4,2,1)
outbreaks_per_month2["No_of_outbreaks"].plot()
plt.title("Nº of Outbreaks per Month")
plt.xlabel("Month")
plt.ylabel("Nº of Outbreaks")

fig.add_subplot(4,2,2)
outbreaks_per_month["Illnesses"].plot()
plt.title("Nº of Illnesses per Month")
plt.xlabel("Month")
plt.ylabel("Nº of Illnesses")

fig.add_subplot(4,2,3) 
outbreaks_per_month["Hospitalizations"].plot()
plt.title("Nº of Hospitalizations per Month")
plt.xlabel("Month")
plt.ylabel("Nº of Hospitalizations")

fig.add_subplot(4,2,4)
outbreaks_per_month["Fatalities"].plot()
plt.title("Nº of Fatalities per Month")
plt.xlabel("Month")
plt.ylabel("Nº of Fatalities")

plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.show()


# __Time Series Decomposition__

# In[63]:


outbreaks_per_year


# In[64]:


from statsmodels.tsa.seasonal import seasonal_decompose
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


# _Nº of Outbreaks_

# In[65]:


outbreaks_ts=outbreaks.groupby(pd.Grouper(key="Date",freq="M")).agg(Nº_of_outbreaks=("Date","count"),
                              Nº_of_Illnesses=("Illnesses","sum"),
                              Nº_of_Hospitalizations=("Hospitalizations","sum"),
                              Nº_of_Fatalities=("Fatalities","sum"))
outbreaks_ts


# In[66]:


decomp=seasonal_decompose(outbreaks_ts["No_of_outbreaks"])


# In[67]:


trend=decomp.trend


# In[68]:


seasonal=decomp.seasonal


# In[69]:


aleatory=decomp.resid


# In[70]:


plt.subplot(4,1,1)
plt.plot(outbreaks_ts["No_of_outbreaks"],label="Outbreaks Original")
plt.legend(loc="best")

plt.subplot(4,1,2)
plt.plot(trend,label="Outbreaks Trend")
plt.legend(loc="best")

plt.subplot(4,1,3)
plt.plot(seasonal,label="Outbreaks and Season Effect")
plt.legend(loc="best")

plt.subplot(4,1,4)
plt.plot(aleatory,label="Outbreaks Random Error")
plt.legend(loc="best")
plt.tight_layout()


# _Illnesses_

# In[71]:


decomp_ill=seasonal_decompose(outbreaks_ts["No_of_Illnesses"])


# In[72]:


trend_ill=decomp_ill.trend


# In[73]:


seasonal_ill=decomp_ill.seasonal


# In[74]:


random_ill=decomp_ill.resid


# In[75]:


plt.subplot(4,1,1)
plt.plot(outbreaks_ts["No_of_Illnesses"],label="Illnesses Original")
plt.legend(loc="best")

plt.subplot(4,1,2)
plt.plot(trend_ill,label="Illnesses Trend")
plt.legend(loc="best")

plt.subplot(4,1,3)
plt.plot(seasonal_ill,label="Illnesses and Season Effect")
plt.legend(loc="best")

plt.subplot(4,1,4)
plt.plot(random_ill,label="Illnesses Random Error")
plt.legend(loc="best")
plt.tight_layout()


# _Hospitalizations_

# In[76]:


decomp_hosp=seasonal_decompose(outbreaks_ts["No_of_Hospitalizations"])
trend_hosp=decomp_hosp.trend
seasonal_hosp=decomp_hosp.seasonal
random_hosp=decomp_hosp.resid


# In[77]:


plt.subplot(4,1,1)
plt.plot(outbreaks_ts["No_of_Hospitalizations"],label="Hospitalizations Original")
plt.legend(loc="best")

plt.subplot(4,1,2)
plt.plot(trend_hosp,label="Hospitalizations Trend")
plt.legend(loc="best")

plt.subplot(4,1,3)
plt.plot(seasonal_hosp,label="Hospitalizations and Season Effect")
plt.legend(loc="best")

plt.subplot(4,1,4)
plt.plot(random_hosp,label="Hospitalizations Random Error")
plt.legend(loc="best")
plt.tight_layout()


# _Fatalities_

# In[78]:


decomp_fatal=seasonal_decompose(outbreaks_ts["No_of_Fatalities"])
trend_fatal=decomp_fatal.trend
seasonal_fatal=decomp_fatal.seasonal
random_fatal=decomp_fatal.resid


# In[79]:


plt.subplot(4,1,1)
plt.plot(outbreaks_ts["No_of_Fatalities"],label="Fatalities Original")
plt.legend(loc="best")

plt.subplot(4,1,2)
plt.plot(trend_fatal,label="Fatalities Trend")
plt.legend(loc="best")

plt.subplot(4,1,3)
plt.plot(seasonal_fatal,label="Ftalities Seasonal Effect")
plt.legend(loc="best")

plt.subplot(4,1,4)
plt.plot(random_fatal,label="Fatalities Random Error")
plt.legend(loc="best")
plt.tight_layout()


# __Time Serie Prevision (ARIMA Method)__

# In[80]:


outbreaks_ts


# In[81]:


#Crete the training set and test set
training_set=outbreaks_ts.loc[outbreaks_ts.index<"2015-01-01"]
test_set=outbreaks_ts.loc[outbreaks_ts.index>="2015-01-01"]


# In[82]:


training_set


# In[83]:


test_set


# In[84]:


fig, ax = plt.subplots(figsize=(15, 5))
training_set["No_of_outbreaks"].plot(ax=ax, label='Training Set')
plt.title('Data Train/Test Split Nº of Outbreaks')
test_set["No_of_outbreaks"].plot(ax=ax, label='Test Set')
ax.axvline('01-01-2015', color='black', ls='--')
ax.legend(['Training Set', 'Test Set'])
plt.show()


# In[85]:


# crete the model
stepwise_model=auto_arima(outbreaks_ts["No_of_outbreaks"],start_q=1,start_d= 0, start_P=0, max_p=6, max_q=6, m=12, seasonal=True, trace=True, stepwise=True)


# In[86]:


print(stepwise_model.aic())


# In[87]:


#Training the model with the training set
stepwise_model.fit(training_set["No_of_outbreaks"])


# In[88]:


# See the model performence (Make the predictions with the test set


# In[89]:


future_forecast = stepwise_model.predict(n_periods=13)
future_forecast


# In[90]:


future_forecast = pd.DataFrame(future_forecast,index = test_set.index,columns=["Nº_of_outbreaks"])


# In[91]:


future_forecast


# In[92]:


pd.concat([test_set["No_of_outbreaks"],future_forecast],axis=1).plot()


# In[93]:


#Mean Squared Error
from sklearn.metrics import mean_squared_error
np.sqrt(mean_squared_error(test_set["No_of_outbreaks"],future_forecast["Nº_of_outbreaks"]))


# In[94]:


# calculate error
np.abs(test_set["No_of_outbreaks"],future_forecast["Nº_of_outbreaks"])


# In[95]:


# Worst and Best Predictions
test_set["error"]=np.abs(test_set["No_of_outbreaks"],future_forecast["Nº_of_outbreaks"])
test_set["date"]=test_set.index.date


# In[96]:


#Worst Prediction
test_set.groupby("date")["error"].mean().sort_values(ascending=False).head(5)


# In[97]:


#Best Prediction
test_set.groupby("date")["error"].mean().sort_values(ascending=True).head(5)


# In[98]:


# Make a predictio for the next 5 years
stepwise_model.predict(n_periods=65)


# In[99]:


next5years=pd.DataFrame(stepwise_model.predict(n_periods=65),columns=["Nº_of_outbreaks"])
next5years


# In[100]:


plt.style.use("ggplot") 
next5years.plot()
plt.title("Nº of Outbreaks for Next 5 Years")
plt.xlabel("Year")
plt.ylabel("Nº of Outbreaks")
plt.legend([],[], frameon=False)
plt.show()


# # Second Goal:
# > What contamination has been responsible for the most ilnesses, hospitalizations and deaths?

# In[101]:


outbreaks.drop(columns="Month",inplace=True)
outbreaks


# In[102]:


outbreaks["Species"].value_counts()


# In[103]:


species=outbreaks.groupby("Species").agg(Nº_of_Outbreaks=("Species","count"),
                                        Nº_of_Illnesses=("Illnesses","sum"),
                                         Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                         Nº_of_Fatalities=("Fatalities","sum"))
species


# __Top 10 Species and Most Outbreaks__

# In[104]:


species_top_10_outbreaks=species.sort_values(by="No_of_Outbreaks",ascending=False).head(10)
species_top_10_outbreaks.drop(columns=["No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
species_top_10_outbreaks


# In[105]:


plt.figure(figsize=(8,5))
species_top_10_outbreaks.plot(kind="bar")
plt.title("Top 10 Species and Outbreaks")
plt.xlabel("Specie")
plt.ylabel("Nº of Outbreaks")
plt.legend([],[], frameon=False)
plt.show()


# In[106]:


plt.figure(figsize = (10, 5))
sns.heatmap(species_top_10_outbreaks.T,cmap='RdYlGn_r',annot=False,fmt='2.0f')
plt.title("Top 10 Species and Nº of Outbreaks",fontsize=18)
plt.show()


# __Top 10 Species and Most Illnesses__

# In[107]:


species_top_10_ill=species.sort_values(by="No_of_Illnesses",ascending=False).head(10)
species_top_10_ill.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
species_top_10_ill


# In[108]:


plt.figure(figsize=(8,5))
species_top_10_ill["No_of_Illnesses"].plot(kind="bar")
plt.title("Top 10 Species and Nº of Illnesses")
plt.xlabel("Specie")
plt.ylabel("Nº of Illnesses")
plt.show()


# In[109]:


plt.figure(figsize = (10, 5))
sns.heatmap(species_top_10_ill.T,cmap='RdYlGn_r',annot=False,fmt='2.0f')
plt.title("Top 20 Species and Nº of Illnesses",fontsize=18)
plt.show()


# __Top 10 Species and Most Hospitalizations__

# In[110]:


species_top_10_hosp=species.sort_values(by="No_of_Hospitalizations",ascending=False).head(10)
species_top_10_hosp.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Fatalities"],inplace=True)
species_top_10_hosp


# In[111]:


plt.figure(figsize=(8,5))
species_top_10_hosp["No_of_Hospitalizations"].plot(kind="bar")
plt.title("Top 10 Species and Nº of Hospitalizations")
plt.xlabel("Species")
plt.ylabel("Nº of Hospitalizations")
plt.show()


# In[112]:


plt.figure(figsize = (10, 5))
sns.heatmap(species_top_10_hosp.T,cmap='RdYlGn_r',annot=False,fmt='2.0f')
plt.title("Top 10 Species and Nº of Hospitalizations",fontsize=18)
plt.show()


# __Top 10 Species and Most Fatalities__

# In[113]:


species_top_10_fatal=species.sort_values(by="No_of_Fatalities",ascending=False).head(10)
species_top_10_fatal.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Hospitalizations"],inplace=True)
species_top_10_fatal


# In[114]:


plt.figure(figsize=(8,5))
species_top_10_fatal.plot(kind="bar")
plt.title("Top 10 Species and Nº of Fatalities")
plt.xlabel("Species")
plt.ylabel("Nº of Fatalities")
plt.show()


# In[115]:


plt.figure(figsize = (10, 5))
sns.heatmap(species_top_10_fatal.T,cmap='RdYlGn_r',annot=False,fmt='2.0f')
plt.title("Top 10 Species and Nº of Fatalities",fontsize=18)
plt.show()


# In[116]:


species


# # Third Goal:
# > Which State for food preparation poses the greatest risk of foodborne illness?

# __Nº of Outbreaks Per State__

# In[117]:


outbreaks


# In[118]:


states=outbreaks.groupby("State").agg(Nº_of_Outbreaks=("State","count"),
                                      Nº_of_Ilnesses=("Illnesses","sum"),
                                      Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                      Nº_of_Fatalities=("Fatalities","sum"))
states


# __States with most outbreaks__

# In[119]:


plt.figure(figsize=(15,8))
plt.style.use("ggplot")
states["No_of_Outbreaks"].plot(kind="bar")
plt.title("Nº of Outbreaks per State 1998-2015")
plt.xlabel("State")
plt.ylabel("Nº of Outbreaks")
plt.show()


# In[120]:


# Choropleth Map
import chart_studio.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs,init_notebook_mode,plot,iplot


# In[121]:


outbreaks


# In[122]:


#Create "Code" Column
outbreaks["State"].value_counts()


# In[123]:


# Clean "state" column 
outbreaks["State"].replace("Washington DC","Washington",inplace=True)


# In[124]:


multistate=outbreaks[outbreaks["State"]=="Multistate"]


# In[125]:


outbreaks_map=outbreaks.drop(multistate.index)


# In[126]:


outbreaks_map["State"].value_counts()


# In[127]:


def code(state):
    if state=="Florida":
        return "FL"
    elif state=="California":
        return "CA"
    elif state=="Ohio":
        return "OH"
    elif state=="Illinois":
        return "IL"
    elif state=="New York":
        return "NY"
    elif state=="Michigan":
        return "MI"
    elif state=="Minnesota":
        return "MN"
    elif state=="Washington":
        return "WA"
    elif state=="Maryland":
        return "MD"
    elif state=="Colorado":
        return "CO"
    elif state=="Oregon":
        return "OR"
    elif state=="Pennsylvania":
        return "PA"
    elif state=="Wisconsin":
        return "WI"
    elif state=="Georgia":
        return "GA"
    elif state=="Texas":
        return "TX"
    elif state=="Kansas":
        return "KS"
    elif state=="Hawaii":
        return "HI"
    elif state=="Tennessee":
        return "TN"
    elif state=="Arizona":
        return "AZ"
    elif state=="Virginia":
        return "VA"
    elif state=="Alabama":
        return "AL"
    elif state=="Connecticut":
        return "CT"
    elif state=="Massachusetts":
        return "MA"
    elif state=="New Jersey":
        return "NJ"
    elif state=="North Carolina":
        return "NC"
    elif state=="Maine":
        return "ME"
    elif state=="Iowa":
        return "IA"
    elif state=="South Carolina":
        return "SC"
    elif state=="Indiana":
        return "IN"
    elif state=="Missouri":
        return "MO"
    elif state=="Utah":
        return "UT"
    elif state=="Alaska":
        return "AK"
    elif state=="Idaho":
        return "ID"
    elif state=="Puerto Rico":
        return "PR"
    elif state=="North Dakota":
        return "ND"
    elif state=="Louisiana":
        return "LA"
    elif state=="New Hampshire":
        return "NH"
    elif state=="Nevada":
        return "NV"
    elif state=="Rhode Island":
        return "RI"
    elif state=="Oklahoma":
        return "OK"
    elif state=="Wyoming":
        return "WY"
    elif state=="New Mexico":
        return "NM"
    elif state=="Arkansas":
        return "AR"
    elif state=="Kentucky":
        return "KY"
    elif state=="Mississippi":
        return "MS"
    elif state=="Vermont":
        return "VT"
    elif state=="Montana":
        return "MT"
    elif state=="West Virginia":
        return "WV"
    elif state=="Nebraska":
        return "NE"
    elif state=="South Dakota":
        return "SD"
    elif state=="Guam":
        return "GU"
    elif state=="Delaware":
        return "DE"   


# In[128]:


outbreaks_map["State Code"]=outbreaks_map["State"].apply(code)


# In[129]:


outbreaks_map


# In[130]:


# Create the Data Frame
state_map=outbreaks_map.groupby("State Code").agg(Nº_of_Outbreaks=("Date","count"),
                                                Nº_of_Illnesses=("Illnesses","sum"),
                                                Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                                Nº_of_Fatalities=("Fatalities","sum"))
state_map


# In[131]:


state_map.reset_index(inplace=True)
state_map


# In[132]:


# Create a "Text" column
state_map["text"]= state_map["State Code"].astype(str)+ "<br>" +   "Nº of Outbreaks"+ state_map["No_of_Outbreaks"].astype(str) +  "<br>" + "Nº of Illnesses"+ state_map["No_of_Illnesses"].astype(str) + "<br>" + "Nº of Hospitalizaions"+ state_map["No_of_Hospitalizations"].astype(str) + "<br>" + "Nº of Fatalities"+ state_map["No_of_Fatalities"].astype(str)


# In[133]:


state_map


# In[134]:


data=dict(type="choropleth",
          colorscale="temps",
          locations=state_map["State Code"],
          locationmode="USA-states",
          z=state_map["No_of_Outbreaks"],
          text=state_map["text"],
          marker=dict(line=dict(color="rgb(255,255,255)",width=4)),
          colorbar={"title":"Nº of Outbreaks"})


# In[135]:


layout=dict(title="Foodborne Disseases by State",
            geo=dict(scope="usa",
            showlakes=True,
            lakecolor="rgb(85,173,240)"))


# In[136]:


choromap2=go.Figure(data=[data],layout=layout)


# In[137]:


iplot(choromap2)


# # Fourth Goal
# 
# > Which Food caused the higher number of outbreaks, hospitalizaions, illnesses and fatalities
# 
# >Which Ingredient caused the higher number of outbreaks, hospitalizaions, illnesses and fatalities
# 
# >Which location is associated to the higher number of outbreaks, hospitalizaions, illnesses and fatalities

# In[138]:


outbreaks


# __Which Food caused the higher number of outbreaks, hospitalizaions, illnesses and fatalities__

# In[139]:


outbreaks["Food"].value_counts()


# In[140]:


food=outbreaks.groupby("Food").agg(Nº_of_Outbreaks=("Date","count"),
                                   Nº_of_Illnesses=("Illnesses","sum"),
                                   Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                   Nº_of_Fatalities=("Fatalities","sum"))
food


# In[141]:


food_outbreaks=food.sort_values(by="No_of_Outbreaks",ascending=False)


# In[142]:


top10_food_outbreaks=food_outbreaks.nlargest(10,"No_of_Outbreaks")
top10_food_outbreaks.drop(columns=["No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
top10_food_outbreaks


# In[143]:


food_outbreaks=food.sort_values(by="No_of_Illnesses",ascending=False)
top10_food_illnesses=food_outbreaks.nlargest(10,"No_of_Illnesses")
top10_food_illnesses.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
top10_food_illnesses


# In[144]:


food_outbreaks=food.sort_values(by="No_of_Hospitalizations",ascending=False)
top10_food_hosp=food_outbreaks.nlargest(10,"No_of_Hospitalizations")
top10_food_hosp.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Fatalities"],inplace=True)
top10_food_hosp


# In[145]:


food_outbreaks=food.sort_values(by="No_of_Fatalities",ascending=False)
top10_food_fatal=food_outbreaks.nlargest(10,"No_of_Fatalities")
top10_food_fatal.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Hospitalizations"],inplace=True)
top10_food_fatal


# In[146]:


fig=plt.figure(figsize=(12,30))

fig.add_subplot(4,2,1)
plot=top10_food_outbreaks["No_of_Outbreaks"].plot(kind="bar")
plt.title("Nº of Outbreaks by Food")
plt.xlabel("Food")
plt.ylabel("Nº of Outbreaks")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center", va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,2)
plot=top10_food_illnesses["No_of_Illnesses"].plot(kind="bar")
plt.title("Nº of Illnesses by Food")
plt.xlabel("Food")
plt.ylabel("Nº of Illnesses")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,3)
plot=top10_food_hosp["No_of_Hospitalizations"].plot(kind="bar")
plt.title("Nº of Hospitalizations by Food")
plt.ylabel("Nº of Hospitalizations")
plt.xlabel("Food")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,4)
plot=top10_food_fatal["No_of_Fatalities"].plot(kind="bar")
plt.title("Nº of Fatalities by Food")
plt.ylabel("Nº of Fatalities")
plt.xlabel("Food")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.tight_layout()
plt.show()
plt.show()


# __Which Ingredient caused the higher number of outbreaks, hospitalizaions, illnesses and fatalities__

# In[147]:


outbreaks


# In[148]:


outbreaks["Ingredient"].value_counts()


# In[149]:


ingredient=outbreaks.groupby("Ingredient").agg(Nº_of_Outbreaks=("Date","count"),
                                              Nº_of_Illnesses=("Illnesses","sum"),
                                              Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                              Nº_of_Fatalities=("Fatalities","sum"))
ingredient


# In[150]:


top10_ingredient_outbreaks=ingredient.nlargest(10,"No_of_Outbreaks")
top10_ingredient_outbreaks.drop(columns=["No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
top10_ingredient_outbreaks


# In[151]:


top10_ingredient_illnesses=ingredient.nlargest(10,"No_of_Illnesses")
top10_ingredient_illnesses.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
top10_ingredient_illnesses


# In[152]:


top10_ingredient_hosp=ingredient.nlargest(10,"No_of_Hospitalizations")
top10_ingredient_hosp.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Fatalities"],inplace=True)
top10_ingredient_hosp


# In[153]:


top10_ingredient_fatal=ingredient.nlargest(10,"No_of_Fatalities")
top10_ingredient_fatal.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Illnesses"],inplace=True)
top10_ingredient_fatal


# In[154]:


fig=plt.figure(figsize=(15,20))

fig.add_subplot(4,2,1)
plot=top10_ingredient_outbreaks["No_of_Outbreaks"].plot(kind="bar")
plt.title("Nº of Outbreaks by Ingredient")
plt.xlabel("Ingredient")
plt.ylabel("Nº of Outbreaks")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,2)
plot=top10_ingredient_illnesses["No_of_Illnesses"].plot(kind="bar")
plt.title("Nº of Illnesses by Ingredient")
plt.xlabel("Ingredient")
plt.ylabel("Nº of Illnesses")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,3)
plot=top10_ingredient_hosp["No_of_Hospitalizations"].plot(kind="bar")
plt.title("Nº of Hospitalizations by Ingredient")
plt.ylabel("Nº of Hospitalizations")
plt.xlabel("Ingredient")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,4)
plot=top10_ingredient_fatal["No_of_Fatalities"].plot(kind="bar")
plt.title("Nº of Fatalities by Ingredient")
plt.ylabel("Nº of Fatalities")
plt.xlabel("Ingredient")
for i in plot.patches:
    plot.annotate(i.get_height(),
                 (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")
    
    
plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.tight_layout()
plt.show()


# __Which Contaminant is associated to fin fish, egg and cantalouque__

# In[155]:


outbreaks.head()


# _Fin Fish_

# In[156]:


fin_fish=outbreaks[outbreaks["Ingredient"]=="Fin Fish"]
fin_fish["Species"].value_counts()


# In[157]:


fin_fish=fin_fish.groupby("Species").agg(Nº_of_Outbreaks=("Date","count"),
                                                  Nº_of_Illnesses=("Illnesses","sum"),
                                                  Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                                  Nº_of_Fatalities=("Fatalities","sum"))
fin_fish


# In[158]:


fin_fish_outbreaks=fin_fish.nlargest(5,"No_of_Outbreaks")
fin_fish_outbreaks.drop(columns=["No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
fin_fish_outbreaks


# In[159]:


fin_fish.reset_index(inplace=True)


# In[160]:


fin_fish["Species"]=fin_fish["Species"].replace("Salmonella enterica; Salmonella enterica; Salmonella enterica; Salmonella enterica; Salmonella enterica; Salmonella enterica","Salmonella enterica")


# In[161]:


fin_fish.set_index("Species",inplace=True)


# In[162]:


fin_fish_ill=fin_fish.nlargest(5,"No_of_Illnesses")
fin_fish_ill.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
fin_fish_ill


# In[163]:


fin_fish_hosp=fin_fish.nlargest(5,"No_of_Hospitalizations")
fin_fish_hosp.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Fatalities"],inplace=True)
fin_fish_hosp


# In[164]:


fin_fish_fatal=fin_fish.nlargest(5,"No_of_Fatalities")
fin_fish_fatal.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Illnesses"],inplace=True)
fin_fish_fatal


# In[165]:


fig=plt.figure(figsize=(15,25))

fig.add_subplot(4,2,1)
plot=fin_fish_outbreaks["No_of_Outbreaks"].plot(kind="bar")
plt.title("Fin Fish Contaminant and Outbreaks")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Outbreaks")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,2)
plot=fin_fish_ill["No_of_Illnesses"].plot(kind="bar")
plt.title("Fin Fish Contaminant and Illnesses")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Illnesses")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,3)
plot=fin_fish_hosp["No_of_Hospitalizations"].plot(kind="bar")
plt.title("Fin Fish Contaminant and Hospitalizations")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Hospitalizations")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

fig.add_subplot(4,2,4)
plot=fin_fish_fatal["No_of_Fatalities"].plot(kind="bar")
plt.title("Fin Fish Contaminant and Fatalities")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Fatalities")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.tight_layout()
plt.show()            


# _Egg_

# In[166]:


outbreaks.head()


# In[167]:


egg=outbreaks[outbreaks["Ingredient"]=="Egg"]
egg


# In[168]:


egg=egg.groupby("Species").agg(Nº_of_Outbreaks=("Date","count"),
                              Nº_of_Illnesses=("Illnesses","sum"),
                              Nº_of_Hospitalizations=("Hospitalizations","sum"),
                              Nº_of_Fatalities=("Fatalities","sum"))
egg


# In[169]:


egg_outbreaks=egg.nlargest(5,"No_of_Outbreaks")
egg_outbreaks.drop(columns=["No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
egg_outbreaks


# In[170]:


egg_ill=egg.nlargest(5,"No_of_Illnesses")
egg_ill.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
egg_ill


# In[171]:


egg_hosp=egg.nlargest(5,"No_of_Hospitalizations")
egg_hosp.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Fatalities"],inplace=True)
egg_hosp


# In[172]:


egg_fatal=egg.nlargest(5,"No_of_Fatalities")
egg_fatal.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Hospitalizations"],inplace=True)
egg_fatal


# In[173]:


fig=plt.figure(figsize=(15,25))

fig.add_subplot(3,2,1)
plot=egg_outbreaks["No_of_Outbreaks"].plot(kind="bar")
plt.title("Egg Contaminant and Outbreaks")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Outbreaks")
for i in plot.patches:
    plot.annotate(i.get_height(),
                     (i.get_x()+i.get_width()/2,i.get_height()),
                      ha="center",va="baseline",fontsize=7,
                      color="black",xytext=(0,1),
                      textcoords="offset points")


fig.add_subplot(3,2,2)
plot=egg_ill["No_of_Illnesses"].plot(kind="bar")
plt.title("Egg Contaminant and Illnesses")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Illnesses")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                   ha="center",va="baseline",fontsize=7,
                   color="black",xytext=(0,1),
                   textcoords="offset points")

fig.add_subplot(3,2,3)
plot=egg_hosp["No_of_Hospitalizations"].plot(kind="bar")
plt.title("Egg Contaminant and Hospitalizations")
plt.xlabel("Contaminant")
plt.ylabel("Nº of Hospitalizations")
for i in plot.patches:
    plot.annotate(i.get_height(),
                 (i.get_x()+i.get_width()/2,i.get_height()),
                  ha="center",va="baseline",fontsize=7,
                  color="black",xytext=(0,1),
                  textcoords="offset points")

plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.tight_layout()
plt.show()         


# _Cantaloupe_

# In[174]:


outbreaks.head()


# In[175]:


cantaloupe=outbreaks[outbreaks["Ingredient"]=="Cantaloupe"]
cantaloupe


# In[176]:


cantaloupe=cantaloupe.groupby("Species").agg(Nº_of_Outbreaks=("Date","count"),
                                             Nº_of_Illnesses=("Illnesses","sum"),
                                             Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                             Nº_of_Fatalities=("Fatalities","sum"))
cantaloupe


# __Which location is associated to the higher number of outbreaks, hospitalizaions, illnesses and fatalities__

# In[177]:


outbreaks


# In[178]:


outbreaks["Location"].value_counts()


# In[179]:


location=outbreaks.groupby("Location").agg(Nº_of_Outbreaks=("Date","count"),
                                           Nº_of_Illnesses=("Illnesses","sum"),
                                           Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                           Nº_of_Fatalities=("Fatalities","sum"))
location


# In[180]:


location_outbreaks=location.nlargest(10,"No_of_Outbreaks")
location_outbreaks.drop(columns=["No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
location_outbreaks


# In[181]:


location_ill=location.nlargest(10,"No_of_Illnesses")
location_ill.drop(columns=["No_of_Outbreaks","No_of_Hospitalizations","No_of_Fatalities"],inplace=True)
location_ill


# In[182]:


location_hosp=location.nlargest(10,"No_of_Hospitalizations")
location_hosp.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Fatalities"],inplace=True)
location_hosp


# In[183]:


location_fatal=location.nlargest(10,"No_of_Fatalities")
location_fatal.drop(columns=["No_of_Outbreaks","No_of_Illnesses","No_of_Hospitalizations"],inplace=True)
location_fatal


# In[184]:


fig=plt.figure(figsize=(15,20))

fig.add_subplot(4,2,1)
plot=location_outbreaks["No_of_Outbreaks"].plot(kind="bar")
plt.title("Nº of Outbreaks by Location")
plt.ylabel("Nº of Outbreaks")
plt.xlabel("Location")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                   ha="center",va="baseline",fontsize=7,
                   color="black",xytext=(0,1),
                   textcoords="offset points")
    
fig.add_subplot(4,2,2)
plot=location_ill["No_of_Illnesses"].plot(kind="bar")
plt.title("Nº of Illnesses by Location")
plt.ylabel("Nº of Illnesses")
plt.xlabel("Location")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                   ha="center",va="baseline",fontsize=7,
                   color="black",xytext=(0,1),
                   textcoords="offset points")

fig.add_subplot(4,2,3)
plot=location_hosp["No_of_Hospitalizations"].plot(kind="bar")
plt.title("Nº of Hospitalizations by Location")
plt.xlabel("Location")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                   ha="center",va="baseline",fontsize=7,
                   color="black",xytext=(0,1),
                   textcoords="offset points")

fig.add_subplot(4,2,4)
plot=location_fatal["No_of_Fatalities"].plot(kind="bar")
plt.title("Nº of Fatalities by Location")
plt.ylabel("Nº of Fatalities")
plt.xlabel("Location")
for i in plot.patches:
    plot.annotate(i.get_height(),
                  (i.get_x()+i.get_width()/2,i.get_height()),
                   ha="center",va="baseline",fontsize=7,
                   color="black",xytext=(0,1),
                   textcoords="offset points")
    
plt.subplots_adjust(wspace=0.2,hspace=0.4)
plt.tight_layout()
plt.show()


# # Fifth Goal:
# >Which variable is a related to foodborne diseases?
# 
# >Let´s add Meat Consumption Variable
# 
# >Linear Regression Model

# In[185]:


meat=pd.read_csv("meat_consumption_worldwide.csv",parse_dates=["TIME"])


# In[186]:


meat


# In[187]:


outbreaks


# __Prepare the dataset__

# In[188]:


outbreaks_by_year=outbreaks.groupby(pd.Grouper(key="Date",freq="Y")).agg(Nº_of_Outbreaks=("Date","count"),
                                                                        Nº_of_Illnesses=("Illnesses","sum"),
                                                                        Nº_of_Hospitalizations=("Hospitalizations","sum"),
                                                                        Nº_of_Fatalities=("Fatalities","sum"))
outbreaks_by_year.reset_index(inplace=True)
outbreaks_by_year


# In[189]:


outbreaks_by_year["Year"]=outbreaks_by_year["Date"].dt.year


# In[190]:


outbreaks_by_year


# In[191]:


outbreaks_by_year.drop(columns=["Date"],inplace=True)


# In[192]:


outbreaks_by_year=outbreaks_by_year[["Year","No_of_Outbreaks","No_of_Illnesses","No_of_Hospitalizations","No_of_Fatalities"]]
outbreaks_by_year.set_index("Year",inplace=True)


# In[193]:


outbreaks_by_year


# In[194]:


meat["LOCATION"].value_counts()


# In[195]:


meat=meat[meat["LOCATION"]=="USA"]
meat= meat[meat['MEASURE'] == 'THND_TONNE']
meat


# In[196]:


meat["Year"]=meat["TIME"].dt.year


# In[197]:


meat


# In[198]:


meat=meat[["Year","Value"]]
meat.set_index("Year",inplace=True)
meat


# In[199]:


meat= meat.loc[(meat.index >= 1998) & (meat.index <= 2015)]


# In[200]:


meat=meat.rename(columns={"Value":"Meat Consumption (THND_TONNE)"})


# In[201]:


# Merging outbreaks and meat dataset
data=outbreaks_by_year.merge(meat,how="left",left_on="Year",right_on="Year")


# In[202]:


data


# __Person Correlation Coefficient (R^2):__
# > Summarizes the strngth and direction of the linear association between two quantitative vaiables.
# >It works for numeric data.

# Scale of Correlation Coeficient:
# 
# 0<=r<=0.19---Very Low Correlation
# 
# 0.2<=r<=0.59---Low Correlation
# 
# 0.4<=r<=0.59 Moderate Correlation
# 
# 0.6<=r<=0.79---High Correlation
# 
# 0.8<=r<=1.0---Very High Correlation

# In[203]:


data.corr()


# In[204]:


data.reset_index(inplace=True)
data.drop(columns=["Year"],inplace=True)


# In[205]:


plt.figure(figsize=(10,8))
sns.pairplot(data)
plt.show()


# In[206]:


#Scatter Plot
data.plot(kind="scatter",x="No_of_Illnesses",y="No_of_Outbreaks")

#calculate equation for trendline
z=np.polyfit(data["No_of_Illnesses"],data["No_of_Outbreaks"],1)
p=np.poly1d(z)

#add trendline to plot
plt.plot(data["No_of_Illnesses"],p(data["No_of_Illnesses"]))
plt.title("Nº of Outbreaks and Nº of Illnesses Correlation")


# As we can see the there is a Very High Correlation between the No_of_Outbreaks and the No_of_Illnesses. 
# We can say that 93.7126% of the No_of_Outbreaks can be explaned by the No_of_Illnesses.

# __Simple Linear Regression:__
# 
# >X---"No_of_Illnesses"
# 
# >y---"No_of_Outbreaks"

# In[207]:


from sklearn.linear_model import LinearRegression
from yellowbrick.regressor import ResidualsPlot


# In[208]:


data


# In[209]:


X=data.iloc[:,1].values
y=data.iloc[:,0].values


# In[210]:


X=X.reshape(-1, 1)
y=y.reshape(-1,1)


# In[211]:


#Create the Simple Linear Regression Model
regressor=LinearRegression()
regressor.fit(X,y)


# In[212]:


#coefficients
regressor.intercept_


# In[213]:


#slope
regressor.coef_


# In[214]:


plt.scatter(X, y)
plt.plot(X, regressor.predict(X), color = 'red')
plt.title("Real y vs Adjusted y")


# In[218]:


visualizador = ResidualsPlot(regressor)
visualizador.fit(X, y)
visualizador.poof()

As we can see the residuals doesn´t apper to have a normal distribution, so probably we will need to apply other model
# In[216]:


# Let´s do Some Predictions
# Let´s predict the nº of outbreaks if nº of illnesses=200
regressor.predict([[200]])


# In[217]:


# If nº of illnesses=200 the nº of outbreaks will be nearly 278


# In[ ]:




