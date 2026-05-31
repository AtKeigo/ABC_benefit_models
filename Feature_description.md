# Feature description

This file describes the input features required by the released models for predicting dosimetric benefit with ABC compared with FB in left-sided breast cancer radiotherapy.

Two final Gradient Boosting models are provided:
- a 5-feature model for predicting mean heart dose (MHD) reduction >20%
- a 6-feature model for predicting mean left lung dose (MLD) reduction >10%

## Model 1: Prediction of MHD reduction >20%

| Feature | Definition | Unit |
|---|---|---|
| CVR | Cardiopulmonary volume ratio, defined as heart volume divided by total lung volume. | unitless |
| RLV | Right lung volume obtained from planning CT. | cm³ |
| LLV | Left lung volume obtained from planning CT. | cm³ |
| L | Laterality of the heart, defined as the shortest horizontal distance between the center of the heart and the AB line. | cm |
| TLV | Total lung volume, defined as the sum of left and right lung volumes. | cm³ |

## Model 2: Prediction of MLD reduction >10%

| Feature | Definition | Unit |
|---|---|---|
| MaxHD | Maximum heart distance, defined as the maximum perpendicular distance from the cardiac contour to line AD on the axial CT slice at the level of the right diaphragmatic dome. | cm |
| MaxLD | Maximum lung distance, defined as the perpendicular distance from the point where the cardiac contour intersects line AD to the left lung border on the same axial CT slice. | cm |
| AB | Anteroposterior thoracic diameter, defined as the distance from the center of the sternum to the center of the spinal canal cavity on the axial CT slice at the level of the right diaphragmatic dome. | cm |
| TLV | Total lung volume, defined as the sum of left and right lung volumes. | cm³ |
| RLV | Right lung volume obtained from planning CT. | cm³ |
| CD | Transverse thoracic diameter, defined as the line perpendicular to AB through its midpoint, extending to the right and left lung borders on the same axial CT slice. | cm |


## Input format

The column names in the input CSV files must exactly match the feature names listed above. Each row should represent one patient.

Example input files are provided in the `sample_data/` folder.
