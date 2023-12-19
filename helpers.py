import json
import os
import shutil
import time
import pandas as pd
import numpy as np
import pathlib
import geopandas as gpd
import pysal as ps
from pysal.viz.mapclassify import MaxP
import libpysal
from sqlalchemy import create_engine, text
from sklearn import cluster
from sklearn.preprocessing import scale

# Data reading & Processing
app_path = pathlib.Path(__file__).parent.resolve()
data_path = pathlib.Path(__file__).parent.joinpath("data")
geo_json_path = data_path.joinpath("boston-zip-codes.geojson")

# PostgreSQL connection parameters
db_params = {
    "host": os.environ["POSTGRESQL_HOST"],
    "port": os.environ["POSTGRESQL_PORT"],
    "database": os.environ["POSTGRESQL_DATABASE_NAME"],
    "user": os.environ["POSTGRESQL_LOGIN"],
    "password": os.environ["POSTGRESQL_PASSWORD"],
}

# Create a SQLAlchemy engine
engine = create_engine(
    f"postgresql://{db_params['user']}:{db_params['password']}@{db_params['host']}:{db_params['port']}/{db_params['database']}"
)



# Use a SQL query to fetch data directly from the database
schema = "BnB"
query = text(f'SELECT * FROM "{schema}".listing_data;')
with engine.connect() as connection:
    boston_listings = pd.read_sql(query, connection)

#boston_listings = pd.read_sql(query, engine)
review_columns = [c for c in boston_listings.columns if "review_" in c]

# Geojson loading
with open(geo_json_path) as response:
    zc_link = json.load(response)
    # Add id for choropleth layer
    for feature in zc_link["features"]:
        feature["id"] = feature["properties"]["neighbourhood"]

listing_zipcode = boston_listings["neighbourhood_cleansed"].unique()
# Close the SQLAlchemy engine
engine.dispose()

def apply_clustering():
    """
     # Apply KMeans clustering to group zipcodes into categories based on type of houses listed(i.e. property type)
    :return: Dataframe.
    db: scaled proportions of house types by zipcode, use for plotting Choropleth map layer.
    barh_df : scaled proportion of house type grouped by cluster, use for prop type chart and review chart.
    """
    variables = ["bedrooms", "bathrooms", "beds"]
    aves = boston_listings.groupby("neighbourhood_cleansed")[variables].mean(numeric_only=True)
    review_aves = boston_listings.groupby("neighbourhood_cleansed")[review_columns].mean()
    types = pd.get_dummies(boston_listings["property_type"])
    prop_types = types.join(boston_listings["neighbourhood_cleansed"]).groupby("neighbourhood_cleansed").sum()
    prop_types_pct = (prop_types * 100.0).div(prop_types.sum(axis=1), axis=0)
    aves_props = aves.join(prop_types_pct)

    # Standardize a dataset along any axis, Center to the mean and component wise scale to unit variance.
    db = pd.DataFrame(
        scale(aves_props), index=aves_props.index, columns=aves_props.columns
    ).rename(lambda x: str(x))

    # Apply clustering on scaled df
    km5 = cluster.KMeans(n_clusters=5, n_init=10)
    km5cls = km5.fit(db.values)
    # print(len(km5cls.labels_))
    db["cl"] = km5cls.labels_
    # sort by labels since every time cluster is running, label 0-4 is randomly assigned
    db["count"] = db.groupby("cl")["cl"].transform("count")

    db.sort_values("count", inplace=True, ascending=True)
    barh_df = prop_types_pct.assign(cl=km5cls.labels_).groupby("cl").mean()
    
    # Join avg review columns for updating review plot
    db = db.join(review_aves)
    grouped = db.groupby("cl")[review_columns].mean()
    barh_df = barh_df.join(grouped)

    return db.reset_index(), barh_df


def rating_clustering(threshold):
    start = time.time()
    # Explore boundaries/ areas where customers are have similar ratings. Different from
    # predefined number of output regions, it takes target variable(num of reviews, and
    # apply a minimum threshold (5% per region) on it.
    # Bring review columns at zipcode level
    rt_av = boston_listings.groupby("neighbourhood_cleansed")[review_columns].mean().dropna()
    # Regionalization requires building of spatial weights
    zc = gpd.read_file(geo_json_path)
    zrt = zc[["geometry", "neighbourhood"]].join(rt_av, on="neighbourhood").dropna()

    w = libpysal.weights.Queen.from_dataframe(zrt, use_index=True)
    
   # zrt.to_file("tmp")
    #w = libpysal.weights.Queen("tmp/tmp.shp", idVariable="neighbourhood")
    # Remove temp tmp/* we created for spatial weights
   # if os.path.isdir(os.path.join("", "tmp")):
     #   print("removing tmp folder")
     #   shutil.rmtree(os.path.join("", "tmp"))
    # Impose that every resulting region has at least 5% of the total number of reviews
    n_review = (
        boston_listings.groupby("neighbourhood_cleansed")["number_of_reviews"]
        .sum()
        .rename(lambda x: str(x))
        .reindex(zrt["neighbourhood"])
    )
    thr = np.round(int(threshold) / 100 * n_review.sum())
    # Set the seed for reproducibility
    np.random.seed(1234)
    #z = zrt.drop(["geometry", "neighbourhood"], axis=1).values
    z = zrt.drop(["geometry", "neighbourhood"], axis=1).values.flatten()
    # Create max-p algorithm, note that this API is upgraded in pysal>1.11.1
    #maxp = mapclassify.MaxP(connectivity=w, data=z, k=thr, floor=n_review.values[:, None], initial=100)
    maxp = MaxP(z, k=5, initial=1000, seed1=0, seed2=1)
    #maxp = mapclassify.MaxP(w, z, thr, num_perm=99, initial=100)
   # maxp.cinference(nperm=99)
    # p value compared with randomly assigned region
    #p_value = maxp.cpvalue
    #print("p_value:", p_value)
    # Run the test
   # maxp_results = maxp.cI(nperm=99)
    
    # Access the simulated labels
   # sim_labels = maxp_results.sim_labels_
    
    # Assuming zrt["neighbourhood"] is your original neighborhood variable
   # lbls = pd.Series(sim_labels).reindex(zrt["neighbourhood"])
   # lbls = pd.Series(maxp.area2region).reindex(zrt["neighbourhood"])
   # regionalization_df = (
   #     pd.DataFrame(lbls).reset_index().rename(columns={"neighbourhood": "neighbourhood", 0: "cl"})
   # )
    end = time.time()
    # The larger threshold, the longer time it takes for computing
    print(
        "Computing threshold {}%".format(threshold),
        "time cost for clustering: ",
        end - start,
    )
    types = pd.get_dummies(boston_listings["property_type"])
    prop_types = types.join(boston_listings["neighbourhood_cleansed"]).groupby("neighbourhood_cleansed").sum()
    # merged = pd.merge(
    #     prop_types.reset_index(), regionalization_df, on="ZIP5", how="inner"
    # )
    # d_merged = merged.drop(["ZIP5", "cl"], axis=1)
    # prop_types_pct = (d_merged * 100.0).div(d_merged.sum(axis=1), axis=0)
    # pct_d = (
    #     prop_types_pct.assign(cl=merged["cl"], zipcode=merged["neighbourhood_cleansed"])
    #     .groupby("cl")
    #     .mean()
    # )
    # zrt = zrt[review_columns].groupby(lbls.values).mean()
    # joined_prop = pct_d.join(zrt)
    # return regionalization_df, p_value, joined_prop
    return zrt, 1, prop_types
# #
# rating = rating_clustering(5)