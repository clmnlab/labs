 % Free parameters: alpha (learning rate), beta (inverse temperature), 
% sig (width of RBF), tau (sensitivity of reward to error)
tar = [160 -160]; tau = 40; alpha = 0.2;
N = 300; mu_glove = mData; sig{1} = corr(allData');  G = [420 420];

for j=1:N  % Number of trials
 	if j==1
       	u(j,:) = mu_glove';  % Update for mean
        vfunc(1,:,:) = 
    	else
       	for d=1:14 % For each dimension
               vfunc(j,d,:) = squeeze(vfunc(j-1,d,:))+ alpha*gaussian(wspace,u(j-1,d),sig)'.*(rew(j-1)-squeeze(vfunc(j-1,d,:)));  % Update value functions            	        	cumsum_val(d,:)=cumsum(exp(beta*vfunc(j,d,:)))/sum(exp(beta*vfunc(j,d,:)));
		% Calculate distribution using soft-max
            	tem = find(rand-cumsum_val(d,:)<0); 
            	u(j,d) = wspace(tem(1)); % Select action from the distribution
        	end
    end    
    	 XY(j,:) = G.*(u(j,:)*A);                          % Map to cursor position
    	 err(j) = norm(XY(j,:)-tar); % Calculate error
    	 rew(j) = rewfunc(err(j),tau); % Calculate reward for the error
end

figure;
subplot(131); plot(err); 
subplot(132); plot(rew);
subplot(133); for j=1:N mexpl(j) = trace(sig{j});end
plot(mexpl);