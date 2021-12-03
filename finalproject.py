import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from pandas.api.types import is_numeric_dtype
from tensorflow import keras
import matplotlib.pyplot as plt
import pandas as pd
pd.set_option('display.float_format', lambda x: '%.3f' % x)
import altair as alt
import seaborn as sns
import streamlit as st

st.title("research on penguins")


penguins = sns.load_dataset("penguins")
df = pd.DataFrame(penguins)

st.subheader("First, we get the dataset of penguins")
st.markdown("The dataset comes from the website: https://seaborn.pydata.org/introduction.html#composite-views-onto-multivariate-datasets")
st.write(df)

st.subheader("Next step, we try to sort this dataset")
st.markdown("by using 'notna' we delete the NaN data")
df = df[df.notna().all(axis = 1)]
st.write(df)

st.markdown("Observing the new data, I notcie that the data of Torgersen island is much less than other island, which cannot fully show the relationship. Therefore, I delete them and only compare the datas of two other island")
choice = st.slider("choose one island you want to check the number(1: Torgersen, 2:Biscoe, 3:Dream)",0,3)
Dict1 = {0:"(Please choose an island", 1:"the number of datas on Torgersen island is", 2:"the number of datas on Biscoe island is", 3:"the number of datas on Dream island is"}
Tor = (df["island"] == "Torgersen").sum()
Bis = (df["island"] == "Biscoe").sum()
Dre = (df["island"] == "Dream").sum()
Dict2 = {0:"in the slider)",1:Tor, 2:Bis, 3:Dre}
st.write(Dict1[choice], Dict2[choice])
df = df[(df["island"] == "Biscoe")| (df["island"] == "Dream")]
st.markdown("Now, we have a more precise dataset")
st.write(df)

st.markdown("Since the ability of male penguins and female penguins are different, so I decide to saparate them into two groups to compare.")
df_male = df[(df["sex"] == "Male")]
st.write(df_male)

st.markdown("Relationship between bill length and bill depth under **different island** for male")
a = alt.Chart(df_male).mark_circle().encode(
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "island",
    tooltip = ["species","body_mass_g"]
)
st.altair_chart(a)


st.markdown("Relationship between bill length and bill depth under **different species** for male")
b = alt.Chart(df_male).mark_line().encode(
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "species",
    tooltip = ["island","body_mass_g"]
)
st.altair_chart(b)


st.markdown("Relationship between bill length and bill depth under **different level of body mass** for male")
c = alt.Chart(df_male).mark_point().encode(
    x = "bill_length_mm",
    y = "bill_depth_mm",
    color = "body_mass_g",
    tooltip = ["species","island"]
)
st.altair_chart(c)

st.markdown("From the three charts above, we learn that: 1.Penguins on Dream island have greater bill length and bill depth. 2. Penguins of Chinstrap have greater bill length and bill depth.3. More heavey penguins are, less depth they can arrive(It is almost the same for famle penguins).")

st.markdown("Then I try to make a logistic regession line to find the relationship between body mass and {bill length & bill depth}")
X = df[["bill_length_mm","bill_depth_mm"]]
Y = df[["body_mass_g"]]

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression()
clf.fit(X,Y)
coef = clf.coef_
intercept = clf.intercept_
perdict = clf.predict(X)
st.write("From the LogisticRegression, we find out the coefficient:",coef,"and the intercept:",intercept)

st.markdown("Finally, we pick one speice to overfit the data by spotify")
numeric_cols = [c for c in df.columns if is_numeric_dtype(df[c])]
st.markdown("We sort the columns, which are in numeric")
st.write(numeric_cols)
st.markdown("We choose the specie Adelie")
df["is_Adelie"] = df["species"].map(lambda g_list:"Adelie" in g_list)
st.write(df["is_Adelie"])
d = df["is_Adelie"].value_counts()
e = 99/(187+99)
st.write("According to the value of specie Adelie",d,", we can calculate",e,", which means that about 65% of penguins are not Adelie")
X_train = df[numeric_cols]
y_train = df["is_Adelie"]
model = keras.Sequential(
    [
        keras.layers.InputLayer(input_shape = (4,)),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(16, activation="sigmoid"),
        keras.layers.Dense(1,activation="sigmoid")
    ]
)

model.compile(
    loss="binary_crossentropy", 
    optimizer=keras.optimizers.SGD(learning_rate=0.01),
    metrics=["accuracy"],
)

history = model.fit(X_train,y_train,epochs=100, validation_split = 0.2, verbose = False)
fig1, ax = plt.subplots()
ax.plot(history.history['loss'])
ax.plot(history.history['val_loss'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
model.summary()
st.write(fig1)
st.markdown("From the plot above, we can figure out that almost no epoches are similar between the training set and the validation set.")

st.markdown("The following one plot accuracies, and we can also see that there are almost no points overfitting")
fig2, ax = plt.subplots()
ax.plot(history.history['accuracy'])
ax.plot(history.history['val_accuracy'])
ax.set_ylabel('loss')
ax.set_xlabel('epoch')
ax.legend(['train', 'validation'], loc='upper right')
st.write(fig2)
st.markdown("Therefore, I do not think I am overfitting the data")

st.subheader("The following is the code from original website. Through the three plots, we can clearly see the distribution. I think this is perfect and pretty cool.")
st.markdown("Here is the link of the website where the code comes from: https://seaborn.pydata.org/introduction.html#composite-views-onto-multivariate-datasets")
st.sidebar.subheader("Joint plot")
select_box3 = st.sidebar.selectbox(label = "x", options = numeric_cols)
select_box4 = st.sidebar.selectbox(label = "y", options = numeric_cols)
sns.jointplot(x = select_box3, y = select_box4, data = penguins)
st.pyplot()

sns.jointplot(data=penguins, x="flipper_length_mm", y="bill_length_mm", hue="species")
st.pyplot()

sns.pairplot(data=penguins, hue="species")
st.pyplot()
st.markdown("The last thing I want to mention is that since the original code cannot show up on streamlit, I also get help from another webiste: https://www.youtube.com/watch?v=uSEbEataipE")