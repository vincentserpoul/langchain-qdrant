import pandas as pd


MAX_TEXT_LENGTH = 1000  # Maximum num of text characters to use


def auto_truncate(val):
    """Truncate the given text."""
    return val[:MAX_TEXT_LENGTH]


# Load Product data and truncate long text fields
all_prods_df = pd.read_csv("product_data.csv", converters={
    'bullet_point': auto_truncate,
    'item_keywords': auto_truncate,
    'item_name': auto_truncate
})

# Replace empty strings with None and drop
all_prods_df['item_keywords'].replace('', None, inplace=True)

all_prods_df.dropna(subset=['item_keywords'], inplace=True)

# Reset pandas dataframe index
all_prods_df.reset_index(drop=True, inplace=True)

# Num products to use (subset)
NUMBER_PRODUCTS = 10

# Get the first NUMBER_PRODUCTS products
product_metadata = (
    all_prods_df
    .head(NUMBER_PRODUCTS)
    .to_dict(orient='index')
)


# data that will be embedded and converted to vectors
texts = [
    v['item_name'] for k, v in product_metadata.items()
]

# product metadata that we'll store along our vectors
metadatas = list(product_metadata.values())
