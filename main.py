from data_preparation import prepare_data
from model_training import train_model, evaluate_model

def main():
    file_path = '/Users/Isabell/Desktop/BSc/hsr4hci/datasets/r_cra__lp/output/r_cra__lp.hdf'
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