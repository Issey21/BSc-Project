from data_preparation import prepare_data
from model_training import train_model, evaluate_model

# Please following the link to download the dataset, I recommend the r_cra__lp.hdf set. 
# https://edmond.mpg.de/dataset.xhtml?persistentId=doi:10.17617/3.LACYPN#
# 

def main():
    file_path = 'path\to\dataset'
    
    train_loader, val_loader, test_loader, input_size = prepare_data(file_path)

    hidden_sizes = [256, 256]
    output_size = 1
    num_epochs = 15

    # Train the model
    model = train_model(train_loader, val_loader, input_size, hidden_sizes, output_size, num_epochs)

    # Evaluate the model
    evaluate_model(model, test_loader)

if __name__ == "__main__":
    main()
