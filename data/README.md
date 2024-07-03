# Vietnamese News Articles Dataset

## Overview
This dataset consists of Vietnamese news articles collected from various Vietnamese online news portals. The dataset was originally sourced from a MongoDB dump containing over 20 million articles. From this large dataset, our team extracted approximately 162,000 articles categorized into 13 distinct categories.

Link dataset: https://github.com/binhvq/news-corpus

## Sample Data
Here is an example of the original data structure:

```json
{
    "source": "Thanh Niên",
    "title": "Đà Nẵng nghiên cứu tiện ích nhắn tin khi vi phạm đến chủ phương tiện",
    "sapo": "Theo thống kê của Phòng CSGT (PC67, Công an TP.Đà Nẵng), từ ngày 1.1.2016 đến hết tháng 1.2018, PC67 gửi 13.479 lượt thông báo đến chủ phương tiện vi phạm luật Giao thông đường bộ.",
    "body": "<p class=\"body-image\"><img src=\"https://photo-1-baomoi.zadn.vn/w700_r1/18/02/05/4/24858235/1_54839.jpg\"/></p><p class=\"body-text\"><em>Xử l&yacute; phạt nguội qua camera gi&aacute;m s&aacute;t tại Ph&ograve;ng CSGT C&ocirc;ng an TP.Đ&agrave; Nẵng - Nguyễn T&uacute;</em></p>..."
    "id": 24858235,
    "publish": ISODate("2018-02-04T22:15:07Z"),
    "tags": [],
    "keywords": ["Công an TP.Đà Nẵng", "Phan Văn Thương", "Luật giao thông đường bộ", ...],
    "cates": ["Pháp luật"]
}
```
## Dataset Preprocessing

The dataset was preprocessed as follows:

- Extracted two main components: `content` and `category`.
  - `content` includes fields such as `title`, `sapo`, `body`, and `keywords`.
  - `category` represents the classification labels.

- Split into train, test, and validation sets with a ratio of 70%, 15%, and 15% respectively.

## File Structure
- `train_data_162k.json`: JSON file containing the training set.
- `test_data_162k.json`: JSON file containing the test set.
- `val_data_162k.json`: JSON file containing the validation set.
- `processed_data`: Folder containing preprocessed data (including 1 million 4 articles)
- `news_v2`: Folder containing data that has been cleaned of the `body` (including 1 million 4 articles)
- `features_162k_phobertbase.pkl`: File containing features extracted from the PhoBert model