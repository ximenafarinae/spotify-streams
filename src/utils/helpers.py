def save_model(np, model, filepath):
    np.savez(filepath, w_hidden1=model.w_hidden1, b_hidden1=model.b_hidden1,
             w_hidden2=model.w_hidden2, b_hidden2=model.b_hidden2,
             w_output=model.w_output, b_output=model.b_output)