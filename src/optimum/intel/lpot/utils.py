

def model_calibration(self, q_model, dataloader, iterations=1):
    import torch
    assert iterations > 0
    with torch.no_grad():
        for idx, input in enumerate(dataloader):
            _ = q_model(**input)
            if idx >= iterations - 1:
                break

