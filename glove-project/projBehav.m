close all; 
clear all;
% experiment parameters
Gx = 420; Gy = 420; X0 = 0; Y0 = 0;
subjlist = ['GA01';'GA02';'GA03';'GA04';'GA05';'GA06';'GA07'];
R = 7;
S = size(subjlist,1);
for s=1:S
% % % for s = 1:1
    subj = subjlist(s,:);
    perf(s) = calcRegressors(subj,'OFF');
    targetID = perf(s).targetID(1:97*R);
    tID = unique(targetID(1:97));
    A = perf(s).U(:,[1 2])';
    U = perf(s).U;
    Ts = pinv(A)*A;
    Ns = eye(14)-Ts;
    fgdata = perf(s).fgdata;
    idx = find(perf(s).fb(1:97)==1); idx=idx(2:end);
    for r=1:R
        for n=1:97            
            data_t =Ts*squeeze(fgdata(n,r,:,:));
            data_n =Ns*squeeze(fgdata(n,r,:,:));
            pscore = U'*squeeze(fgdata(n,r,:,:));
            pvar = var(pscore');
            var_t(n,r,s) = sum(pvar(1:2))/sum(pvar);
            var_n(n,r,s) = sum(pvar(3:end))/sum(pvar);
            
            
            tidx = find(perf(s).cnt_hit_all(97*(r-1)+n,:)==1);
            if isempty(tidx)
                mfgdata(s,r,n,:) = zeros(1,14);
                sfgdata(s,r,n,:) = zeros(1,14);
            elseif length(tidx)==1
                mfgdata(s,r,n,:) = squeeze(fgdata(n,r,:,tidx))';
                sfgdata(s,r,n,:) =  zeros(1,14);
            else
                mfgdata(s,r,n,:) = mean(squeeze(fgdata(n,r,:,tidx))');
                sfgdata(s,r,n,:) = std(squeeze(fgdata(n,r,:,tidx))');

            end
            
        end

            for t=1:length(tID)
                tid1 = idx(find(targetID(idx) == tID(t)));
                
                var_ttID1(t,:,r,s) = var_t(tid1,r,s);
                var_ntID1(t,:,r,s) = var_n(tid1,r,s);

            end
    end
end
c_code = ['rgbkcy'];


R = 7;
for s=1:S
    for r=1:R
%         cost1(s,:,r,:) = perf(s).v2cost(idx,r,:);
%         cost0(s,:,r,:) = perf(s).v2cost(idx0,r,:);
%         cost(s,:,r,:) = perf(s).v2cost(:,r,:);
        cost(s,:,r,:) = perf(s).avcost(:,r,:);

% cost(s,:,r,:) = sqrt(perf(s).v2cost(:,r,:));

        for k=1:8
            mcost(s,k,r) = squeeze(mean(mean(cost(s,12*(k-1)+2:12*k+1,r,:))))';
            mrew(s,k,r) = mean(perf(s).cnt_hit(97*(r-1)+12*(k-1)+2:97*(r-1)+12*k+1));
            merr(s,k,r) = mean(mean(perf(s).errR(12*(k-1)+2:12*k+1,r,:)));
        end
    end
            rew(s,:,:) = reshape(perf(s).cnt_hit(1:97*R),97,R);
end

rewpcost = reshape(rew./squeeze(sum(cost,4)),S,97*R);
% rewpcost = reshape(squeeze(rew)./squeeze(sum(cost,4)),S,97*R);

mrewpcost = reshape(mrew./mcost,S,8*R);
mperfpcost = reshape((max(max(max(merr)))-merr)./mcost,S,8*R);

figure;
subplot(221);
errorbar(mean(mrewpcost),std(mrewpcost)/sqrt(S));
vline([24.5 48.5])

subplot(222);
errorbar(mean(mrew(:,1:end)),std(mrew(:,1:end))/sqrt(S));
vline([24.5 48.5])

subplot(223);
errorbar(mean(mcost(:,1:end)),std(mcost(:,1:end))/sqrt(S));
vline([24.5 48.5])

subplot(224);
errorbar(mean(merr(:,1:end)),std(merr(:,1:end))/sqrt(S));
vline([24.5 48.5])

for s=1:S
    for r=1:6
        [vec(s,r,:,:) tem]=performPCA(squeeze(cost(s,idx,r,:))-repmat(mean(squeeze(cost(s,idx,r,:))),length(idx),1));
        val(s,r,:) = diag(tem);
    end
end

figure;
for r=1:6 
    errorbar(mean(cumsum(squeeze(val(:,r,:)),2)./repmat(sum(squeeze(val(:,r,:)),2),1,14)),...
        std(cumsum(squeeze(val(:,r,:)),2)./repmat(sum(squeeze(val(:,r,:)),2),1,14))/sqrt(23),'Color',c_code(r));
    hold on;
end

% figure;
% for r=1:4 
%     for t=1:4  
%         tid1 = idx(find(targetID(idx) == tID(t)));
%         subplot(2,2,t);
%         for k=1:3 plot((squeeze(mfgdata(s,r,tid1(k:3:end),:))'),'Color',c_code(k));
%             hold on;ylim([-0.8 1]);
%         end
%     end
% end
% 
% figure;
% for r=1:4 
%     for t=1:4  
%         tid0 = idx0(find(targetID(idx0) == tID(t)));
%         subplot(2,2,t);
%         for k=1:3 plot((squeeze(mfgdata(s,r,tid0(k:3:end),:))'),'Color',c_code(k));
%             hold on;ylim([-0.8 1]);
%         end
%     end
% end
% 

S = size(subjlist,1);
figure;
for i=1:4 
    subplot(2,2,i);
    errorbar(mean(reshape(squeeze(var_ttID1(i,:,:,:)),24*R,S)'),...
        std(reshape(squeeze(var_ttID1(i,:,:,:)),24*R,S)')/sqrt(S));
    ylim([0.3 0.65]);vline([24.5 48.5]);
end
% 
figure;
for i=1:4 
    subplot(4,3,3*(i-1)+k);
    errorbar(mean(reshape(squeeze(var_ntID1(i,k:3:end,:,:)),6*R,S)'),...
        std(reshape(squeeze(var_ntID1(i,k:3:end,:,:)),6*R,S)')/sqrt(S));
        ylim([0.3 0.65]);

end

% figure;
% for i=1:4 
%     for k=1:3
%     subplot(4,3,3*(i-1)+k);
%     errorbar(mean(reshape(squeeze(var_ttID1(i,k:3:end,:,:)),6*R,S)'),...
%         std(reshape(squeeze(var_ttID1(i,k:3:end,:,:)),6*R,S)')/sqrt(S));
%     ylim([0.3 0.65]);
%     end
% end
% % 
% figure;
% for i=1:4 
%     for k=1:3
%     subplot(4,3,3*(i-1)+k);
%     errorbar(mean(reshape(squeeze(var_ntID1(i,k:3:end,:,:)),6*R,S)'),...
%         std(reshape(squeeze(var_ntID1(i,k:3:end,:,:)),6*R,S)')/sqrt(S));
%         ylim([0.3 0.65]);
% 
%     end
% end
% 
