function plotMReward(total_n)    
    for i =1:total_n
        figure()
        subj= extractPResult(i,6);
        plot(subj.left.meanReward)
        hold on
        plot(subj.right.meanReward)
        title(strcat('Subj0',num2str(i)))
        legend('Left','Right')
        hold off
    end
end
        
        