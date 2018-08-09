function rew = rewfunc(err,tau)
rew = exp(-err.^2./(2*tau^2));
