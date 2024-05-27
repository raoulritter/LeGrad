import torch

def load_heatmap(path):
    """
    Load heatmap tensor from a given file path.

    Parameters:
    path (str): Path to the file containing the heatmap tensor.

    Returns:
    torch.Tensor: Loaded heatmap tensor.
    """
    return torch.load(path)

def calculate_heatmap_strength(heatmap, bbox):
    """
    Calculate the sum of heatmap values within the bounding box.

    Parameters:
    heatmap (torch.Tensor): 2D tensor representing the heatmap.
    bbox (tuple): Bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
    float: Sum of heatmap values within the bounding box.
    """
    x_min, y_min, x_max, y_max = bbox
    return torch.sum(heatmap[y_min:y_max, x_min:x_max]).item()

def find_strongest_bounding_box(heatmap, bounding_boxes):
    """
    Find the bounding box where the heatmap is the strongest as a percentage of the bounding box size.

    Parameters:
    heatmap (torch.Tensor): 2D tensor representing the heatmap.
    bounding_boxes (list of tuples): List of bounding box coordinates (x_min, y_min, x_max, y_max).

    Returns:
    tuple: The bounding box with the highest heatmap intensity percentage.
    """
    max_intensity = 0
    best_bbox = None
    
    for bbox in bounding_boxes:
        x_min, y_min, x_max, y_max = bbox
        bbox_area = (x_max - x_min) * (y_max - y_min)
        heatmap_intensity = calculate_heatmap_strength(heatmap, bbox)
        intensity_percentage = heatmap_intensity / bbox_area
        
        if intensity_percentage > max_intensity:
            max_intensity = intensity_percentage
            best_bbox = bbox
            
    return best_bbox

# Example usage
if __name__ == "__main__":
    heatmap_path = 'path_to_heatmap_tensor.pt'
    
    heatmap = load_heatmap(heatmap_path)

    bounding_boxes = [
        (0, 0, 2, 2),
        (1, 1, 3, 3),
        (2, 2, 4, 4)
    ]

    strongest_bbox = find_strongest_bounding_box(heatmap, bounding_boxes)
    print(f"The bounding box with the highest heatmap intensity percentage is: {strongest_bbox}")