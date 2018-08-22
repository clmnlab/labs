function prob = gaussian(q,mu,sig)
y = exp(-(q-mu).^2./(2*sig^2));
prob = y/sum(y);
    
