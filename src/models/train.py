def train_pipeline(pipeline, df_train, features, target):
    """
    Train a sklearn pipeline on training data.

    Args:
        pipeline: sklearn Pipeline (preprocessing + model).
        df_train: Training DataFrame.
        features: List of feature column names.
        target: Target column name.

    Returns:
        Trained sklearn pipeline.
    """
    model = pipeline.fit(df_train[features], df_train[target])
    return model
