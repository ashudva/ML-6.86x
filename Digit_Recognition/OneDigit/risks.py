#Loss function = Hinge Loss
def empirical_risk_hinge (a, theta):
    loss = 0
    for i, z in enumerate(a):
        x = z[1] - np.dot(theta,z[0])
        if x < 0:
            loss += 1 - x
        else:
            loss += max(0,1-x)
    return loss / len(a)

#Loss function = Squared Error
def empirical_risk_squared (a,theta):
    loss = 0
    for i, z in enumerate(a):
        loss += (z[1] - np.dot(theta,z[0]))**2 / 2
    return loss / len(a)
