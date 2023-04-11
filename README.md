# Momentum Trading using Twitter

In this project, we explore momentum trading using twitter. We have cut the repo up into several branches, each representing a part of the project.

They are as follows:
log_reg: in this branch, we explore logistic regression on the twitter data and the news data. We save news data because it performs better.

bert: in this branch, we explore using base bert as a sentiment creator and also using it for keyword detection in a similar manner as the logistic regression

finbert: this is the same as bert, but we use the fin bert model.

twit: this is the branch that we explore using twitter data in. We also built the momentum strategy in this branch, and include several versions of it.

Our datasets are too large to upload to github, but they can be found at the dropbox link here:
https://www.dropbox.com/scl/fo/xgo9z4n720iy00z3ht1mu/h?dl=0&rlkey=ohtcz0dln8qj9vb96c82lcldw

This consists of:
- chart information for amazon and apple
- news articles scraped from NASDAQ
- tweets about several companies from the period of 2015-2020
