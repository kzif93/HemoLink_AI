from src.data_loader import load_geo_series_matrix

def test_loader_with_file(path):
    with open(path, "rb") as f:
        data, labels, metadata = load_geo_series_matrix(f)
        print("Shape:", data.shape)
        print("Labels:", labels[:5])
        print("Metadata preview:")
        print(metadata.head())

# Example:
# test_loader_with_file("data/GSE19151_series_matrix.txt")
