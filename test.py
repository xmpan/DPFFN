import torch
import os
import argparse
from utils.data import MyDataSetVH, read_data_vh
from utils.model import MyLayer


def test_model(model, test_loader, device):
    """
    Run inference on the test dataset and return the predictions and true labels.
    """
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for inputs_hh, inputs_vh, labels in test_loader:
            inputs_hh, inputs_vh, labels = inputs_hh.to(device), inputs_vh.to(device), labels.to(device)

            outputs, _, _, _, _ = model(inputs_hh, inputs_vh)
            predicted = outputs.argmax(dim=1)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


def main():
    parser = argparse.ArgumentParser(description='Test the trained model')
    parser.add_argument('-model', type=str, required=True, help='Path to the trained model file')
    parser.add_argument('-input_hh', type=str, required=True, help='Path to the input folder or file for testing')
    parser.add_argument('-input_vh', type=str, required=True, help='Path to the input folder or file for testing')
    parser.add_argument('-device', type=str, default='cpu', help='Device to run the model (e.g., "cpu" or "cuda")')
    parser.add_argument('-batch', type=int, default=1, help='Batch size for testing')
    parser.add_argument('-cls', type=int, default=10, help='Number of classes in the dataset')
    parser.add_argument('-len', type=int, default=512, help='Sequence length of the input data')
    args = parser.parse_args()
    
    # Load test data
    input_path_hh , input_path_vh= args.input_hh,args.input_vh
    if os.path.isdir(input_path_hh) and os.path.isdir(input_path_vh):
        test_data_h, test_data_v, test_labels = read_data_vh(
            input_path_hh, input_path_vh
        )
    else:
        raise ValueError("Input path should be a directory containing the test data.")

    test_loader = torch.utils.data.DataLoader(
        dataset=MyDataSetVH(test_data_h, test_data_v, test_labels),
        batch_size=args.batch,
        shuffle=False
    )

    # Initialize the model
    d_model, d_ff, d_k, d_v = 100, 512, 64, 64
    n_heads, n_layers_G, n_layers_L = 10, 10, 10
    seq_len, cls = args.len, args.cls
    size_out = 16

    model = MyLayer(args.batch, d_model, n_layers_G, n_layers_L, cls, args.device, d_ff, seq_len, d_k, d_v, n_heads, size_out).to(args.device)


    # Load the trained model weights
    model.load_state_dict(torch.load(args.model, map_location=args.device))

    # Run the test
    predictions, true_labels = test_model(model, test_loader, args.device)

    # Output the results
    print("Sample Index | Predicted Class | True Class")
    print("-------------------------------------------")
    for idx, (pred, true) in enumerate(zip(predictions, true_labels)):
        print(f"{idx + 1:>12} | {pred:>15} | {true:>10}")


if __name__ == '__main__':
    main()
