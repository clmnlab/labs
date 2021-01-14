function animate_pos(subj_idn, session_n, trial_n,LR,animate_time)
    figure()
    total=load_Psession(subj_idn,session_n);
    xSize=total{session_n}.xCenter*2;
    ySize=total{session_n}.yCenter*2;
    [stimPos, XY, Hit] = stimSet(total, session_n,trial_n, rem(trial_n,2), LR);

    radius= 100;
    axis([0 xSize -ySize 0]);
    title(strcat('Hit :',num2str(Hit))) 
    hold on
    
    for i = 1:5
        circle(stimPos(1,i),-stimPos(2,i),radius,'black');
    end
    
    
    curve = animatedline(); 
    for j = 1:length(XY)
        
        addpoints(curve,XY(1,j),-XY(2,j))
        drawnow()
        circle(XY(1,j),-XY(2,j),20,'b');
        pause(animate_time)
    end

    hold off
end

function h = circle(x,y,r,color)
    th = 0:pi/50:2*pi;
    xunit = r * cos(th) + x;
    yunit = r * sin(th) + y;
    h = plot(xunit, yunit,color);
end

function [stimPos, XY, Hit] = stimSet(total, session_n,trial_n, pos, LR)
    switch pos
        case 1
            stimPos=total{session_n}.positionset1;
        case 0
            stimPos=total{session_n}.positionset2;
    end
    switch LR
        case 'L'
            XY=total{session_n}.learnL.XY{trial_n};
            Hit=total{session_n}.learnL.HitTarget(trial_n);
        case 'R'
            XY=total{session_n}.learnR.XY{trial_n};
            Hit=total{session_n}.learnR.HitTarget(trial_n);
    end
end