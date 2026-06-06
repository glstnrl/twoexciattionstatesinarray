clc;
clear all;
close all;

%S=((sin(teta)*cos(-r/c))/r)^2;
%E=((sin(teta)*cos(-r/c))/r)^2 theta;
%B=((sin(teta)*cos(-r/c))/r)^2 phi;
lamda=6*pi*10^(-6);
R=100*lamda;
k=(2*pi)/lamda;
c=2.99792458e8;
w=k*c;
i=1;
C=3.5e-17;
Evec=0;
Bvec=0;
dteta=pi/2;  %console1
dphi=0;
dvec=[sin(dteta)*cos(dphi),sin(dteta)*sin(dphi),cos(dteta)];

N=11;     %console2
endi=2*lamda;
beg=-endi;
aralik=(endi-beg)/(N-1);
    
for phi=0:0.01:2*pi
     for teta=0:0.01:pi
         for d=beg:aralik:endi 
            posS=[R*sin(teta)*cos(phi),R*sin(teta)*sin(phi),R*cos(teta)];
            absR=posS-[d,0,0];
            d1=sqrt(sum(absR.^2));
            absoluteR=absR/d1;
            sinterm=norm(cross(absR,dvec))/d1;
            E1vec=cross(cross(dvec,absoluteR),absoluteR)*sinterm*cos(-(w*d1)/c);
            B1vec=cross(dvec,absoluteR)*sinterm*cos(-(w*d1)/c);
            Evec=Evec+E1vec;
            Bvec=Bvec+B1vec;
         end
         Svec=cross(Evec,Bvec);
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