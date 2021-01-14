function hit=extractHit(subj_n,day_n,run_n)
    subjdat=load_IFdata(subj_n);
    datas0=subjdat{day_n};
    session_dat=datas0{run_n};
    data1st=session_dat.learnR;
    data2nd=session_dat.learnL;
    hit1st=data1st.HitTarget;
    hit2nd=data2nd.HitTarget;
    totalHit=nan(1,length(hit1st)+length(hit2nd));
    m1p1Hit=nan(1,length(hit1st)/2);
    m2p1Hit=nan(1,length(hit2nd)/2);
    m1p2Hit=nan(1,length(hit1st)/2);
    m2p2Hit=nan(1,length(hit2nd)/2);
    i1st=1;
    i2nd=1;
    for i = 1:length(totalHit)
        switch rem(i,4)
            case 3
                m1p2Hit(ceil(i/4))=hit1st(i1st);
                totalHit(i)=hit1st(i1st);
                i1st=i1st+1;
            case 2
                m2p1Hit(ceil(i/4))=hit1st(i2nd);
                totalHit(i)=hit2nd(i2nd);
                i2nd=i2nd+1;
            case 1
                m1p1Hit(ceil(i/4))=hit1st(i1st);
                totalHit(i)=hit1st(i1st);
                i1st=i1st+1;
            case 0
                m2p2Hit(ceil(i/4))=hit2nd(i2nd);
                totalHit(i)=hit2nd(i2nd);
                i2nd=i2nd+1;
        end
    end
    hit.m1p1=m1p1Hit;
    hit.m1p2=m1p2Hit;
    hit.m2p1=m2p1Hit;
    hit.m2p2=m2p2Hit;
    hit.total=totalHit;
end


