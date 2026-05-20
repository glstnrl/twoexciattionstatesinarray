clc;
clear all;
close all;

%S=((sin(teta)*cos(-r/c))/r)^2;
%E=((sin(teta)*cos(-r/c))/r)^2 theta;
%B=((sin(teta)*cos(-r/c))/r)^2 phi;
lamda=6*pi*10^(-6);
R=100*lamda;
k=(2*pi)/lamda;
i=1;
C=1.4*10^(-19);
Evec=0;
Bvec=0;
dteta=pi/4;
dphi=0;
dvec=[sin(dteta)*cos(dphi),sin(dteta)*sin(dphi),cos(dteta)];

for teta=0:0.01:pi    
    for phi=0:0.01:2*pi
        posS=[R*sin(teta)*cos(phi),R*sin(teta)*sin(phi),R*cos(teta)];
            absR1=posS-[0,0,0];
            d1=sqrt(sum(absR1.^2));
            teta1=acos(absR1(3)/d1);
            phi1=atan2(absR1(2),absR1(1));
            sinterm1=norm(cross(absR1,dvec))/d1;
            E1=((sinterm1*cos(-d1*k))/d1);
            B1=((sinterm1*cos(-d1*k))/d1);
            E1vec=[E1*cos(teta1)*cos(phi1),E1*cos(teta1)*sin(phi1),-E1*sin(teta1)];
            B1vec=[-B1*sin(phi1),B1*cos(phi1),0];
            Evec=Evec+E1vec;
            Bvec=Bvec+B1vec;
        for d=0:0.4*lamda:lamda*2
            posS=[R*sin(teta)*cos(phi),R*sin(teta)*sin(phi),R*cos(teta)];
            absR1=posS-[d/2,0,0];
            absR2=posS-[-d/2,0,0];
            d1=sqrt(sum(absR1.^2));
            d2=sqrt(sum(absR2.^2));
            teta1=acos(absR1(3)/d1);
            teta2=acos(absR2(3)/d2);
            phi1=atan2(absR1(2),absR1(1));
            phi2=atan2(absR2(2),absR2(1));
            sinterm1=norm(cross(absR1,dvec))/d1;
            sinterm2=norm(cross(absR2,dvec))/d2;
            E1=((sinterm1*cos(-d1*k))/d1);
            E2=((sinterm2*cos(-d2*k))/d2);
            B1=((sinterm1*cos(-d1*k))/d1);
            B2=((sinterm2*cos(-d2*k))/d2);
            E1vec=[E1*cos(teta1)*cos(phi1),E1*cos(teta1)*sin(phi1),-E1*sin(teta1)];
            E2vec=[E2*cos(teta2)*cos(phi2),E2*cos(teta2)*sin(phi2),-E2*sin(teta2)];
            B1vec=[-B1*sin(phi1),B1*cos(phi1),0];
            B2vec=[-B2*sin(phi2),B2*cos(phi2),0];
            Evec=Evec+E1vec+E2vec;
            Bvec=Bvec+B1vec+B2vec;
            Svec=cross(Evec,Bvec);
        end
        S=C*sqrt(sum(Svec.^2));
        
            Sshow(i)=S;
            tetavec(i)=teta;
            phivec(i)=phi;
            i=i+1;  
        
        Evec=0;
        Bvec=0;
    end
end

x=R*sin(tetavec).*cos(phivec);
y=R*sin(tetavec).*sin(phivec);
z=R*cos(tetavec);

figure;
scatter3(x,y,z,15,log10(Sshow),'filled');
colorbar;
axis equal;
xlabel('x'); ylabel('y'); zlabel('z');
title('Radiation Scale');
grid on;
