# Yelp Module 3 Findings

## Final Numbers

- City: Philadelphia
- Final row count: 5,852
- Mean stars: 3.56
- Number of AT-LAUNCH features: 15
- Number of AFTER-REVIEWS features: 5

## Top 3 Insights

1. The final Philadelphia restaurant dataset contains 5,852 rows with a mean star rating of 3.56.
2. Restaurants in the middle 50% are open 35-84 hours per week.
3. The average review text length is 551 characters, which gives Module 4/7 useful warm-start signals.

## Feature Signals

Top feature importance signals:
- mean_review_sentiment_score: 0.7912 (AFTER-REVIEWS)
- mean_review_length: 0.0981 (AFTER-REVIEWS)
- review_velocity: 0.0172 (AFTER-REVIEWS)
- hours_open_per_week: 0.0125 (AT-LAUNCH)
- longitude: 0.0121 (AT-LAUNCH)
- latitude: 0.0112 (AT-LAUNCH)
- dist_from_center: 0.0103 (AT-LAUNCH)
- n_categories: 0.0071 (AT-LAUNCH)
- review_count: 0.0071 (AFTER-REVIEWS)
- log_review_count: 0.0069 (AFTER-REVIEWS)

## One Question For Module 4

Can a stronger model, such as Random Forest or Gradient Boosting, beat the simple baseline and reduce mean absolute error below about 0.6 stars?

## Summary Chart

See `reports/figures/top_feature_importance.png` for the top feature-importance chart with AT-LAUNCH and AFTER-REVIEWS coloring.
