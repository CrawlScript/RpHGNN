# Directories
DOWNLOADS_DIR="./downloads"
DATASETS_DIR="./datasets"

mkdir $DOWNLOADS_DIR 
mkdir $DATASETS_DIR

wget -P $DOWNLOADS_DIR https://github.com/CrawlScript/gnn_datasets/raw/master/HGB/ACM.zip 
wget -P $DOWNLOADS_DIR https://github.com/CrawlScript/gnn_datasets/raw/master/HGB/DBLP.zip 
wget -P $DOWNLOADS_DIR https://github.com/CrawlScript/gnn_datasets/raw/master/HGB/Freebase.zip 
wget -P $DOWNLOADS_DIR https://github.com/CrawlScript/gnn_datasets/raw/master/HGB/IMDB.zip 


# Ensure the datasets directory exists
mkdir -p $DATASETS_DIR

# Unzip each file
unzip $DOWNLOADS_DIR/ACM.zip -d $DATASETS_DIR
unzip $DOWNLOADS_DIR/DBLP.zip -d $DATASETS_DIR
unzip $DOWNLOADS_DIR/Freebase.zip -d $DATASETS_DIR
unzip $DOWNLOADS_DIR/IMDB.zip -d $DATASETS_DIR
