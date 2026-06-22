# Processed Dataset Dictionary

File: `person_year_dataset.csv`

| Column | Description |
| --- | --- |
| `PERSON_ID` | Synthetic resident identifier used for grouped cross-validation. |
| `YEAR_T` | Predictor year. Outcomes are measured in the following year. |
| `SEXO` | Sex as encoded in the original institutional dataset. |
| `EDAD_T` | Age at predictor year. |
| `TPO_INSTIT_T` | Institutionalization-time variable at predictor year, as encoded in the source data. |
| `AYUDA_HABITUAL_T` | Habitual aid/support variable at predictor year, as encoded in the source data. |
| `NUM_CAIDA_T` | Number of falls recorded in predictor year. |
| `FALL_T` | Indicator for at least one fall in predictor year. |
| `FALLS_PREV_2Y_T` | Total number of falls across the current and previous observed year. |
| `FALLS_PREV_3Y_T` | Total number of falls across the current and previous two observed years. |
| `CAIDAS_ACUM_T` | Cumulative number of falls up to and including predictor year. |
| `EVER_FALLEN_T` | Indicator for any fall up to and including predictor year. |
| `OBSERVED_YEARS_T` | Number of observed annual records up to and including predictor year. |
| `MEAN_FALLS_OBSERVED_T` | Mean annual falls across observed years up to predictor year. |
| `TINETTI_LAST_T` | Most recent prior Tinetti score available at or before predictor year. |
| `BERG_LAST_T` | Most recent prior Berg score available at or before predictor year. |
| `HAS_FUNCTIONAL_T` | Indicator for availability of a prior functional assessment. |
| `YEARS_SINCE_FUNCTIONAL_T` | Number of years between predictor year and most recent prior functional assessment. |
| `NUM_CAIDA_NEXT` | Number of falls recorded in the following year. |
| `FALL_NEXT_YEAR` | Indicator for at least one fall in the following year. |
| `RECURRENT_FALL_NEXT_YEAR` | Indicator for two or more falls in the following year. |
| `HIGH_BURDEN_FALL_NEXT_YEAR` | Indicator for four or more falls in the following year. |
| `DEATH_AFTER_FALL_NEXT_YEAR` | Indicator for death after a fall in the following year, as recorded in the source data. |

The dataset contains no direct names. Because this remains row-level health-related information from a small cohort, public release requires final institutional authorization.
