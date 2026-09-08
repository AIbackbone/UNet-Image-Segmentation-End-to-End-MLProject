from utils.logger import get_logger
from model.unet import UNet
from configs.config import CFG

LOG = get_logger('Unet')

def evaluate(self):
    """Predicts results for the test dataset"""

    predictions = []
    LOG.info('Predicting segmentation map for test dataset')

    for image, mask in self.test_dataset:
        LOG.debug(f'Predicting segmentation map {image}')
        predictions.append(self.model.predict(image))
    return predictions

if __name__ == "__main__":
    unet = UNet(CFG)

    unet.load_data()
    unet.build()
    evaluate(unet)