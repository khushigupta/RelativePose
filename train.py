from siamese_model import match_model, pose_model


input_shape = (None, 64, 64, 3)
match = match_model(input_shape)
match.compile(optimizer='sgd',
              loss=['mean_squared_error'],
              metrics=['accuracy'])



input_shape = (None, 224, 224, 3)
pose = pose_model(input_shape)
pose.compile(optimizer='sgd',
             loss=['mean_squared_error', 'mean_squared_error'],
             metrics=['accuracy'],
             loss_weights=[1., 1])






