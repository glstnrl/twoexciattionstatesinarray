clc;
clear all;
close all;

%S=((sin(teta)*cos(-r/c))/r)^2;
%E=((sin(teta)*cos(-r/c))/r)^2 theta;
%B=((sin(teta)*cos(-r/c))/r)^2 phi;
lamda=6*pi*10^(-6);
R=100*lamda;
k=(2*pi)/lamda;
Evec=0;
Bvec=0;
dteta=0;   %console1
dphi=0;
dvec=[sin(dteta)*cos(dphi),sin(dteta)*sin(dphi),cos(dteta)];

hbar=1.054571817e-34;
eps=8.8541878128e-12;
c=2.99792458e8;
mu=2.54e-29;
C=3.5e-17;
w=k*c;
rvec=[1,0,0];
mudotr=dot(rvec,dvec);

N=41;     %console2
endi=lamda;
beg=-endi;
aralik=(endi-beg)/(N-1);
modofsystem=1;

H=zeros(N,N);
Omega=zeros(N,N);
Gamma=zeros(N,N);
r=zeros(N,N);

for i=1:N     
    for j=1:N 
        ri=aralik*i;
        rj=aralik*j;
        r(i,j)=abs(rj-ri);
        if i==j
            Gamma(i,j)=((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3);
        end
            if i~=j  
            Omega(i,j)=(3/4)*((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)*(((-(1-mudotr^2)*cos(k*r(i,j)))/(k*r(i,j)))+((1-3*mudotr^2)*((sin(k*r(i,j)))/(k*r(i,j))^2+cos(k*r(i,j))/(k*r(i,j))^3)));
            Gamma(i,j)=(3/2)*((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)*((1-mudotr^2)*(sin(k*r(i,j))/(k*r(i,j)))+(1-3*mudotr^2)*(cos(k*r(i,j))/(k*r(i,j))^2-sin(k*r(i,j))/(k*r(i,j))^3));
            end
      end
end
H=Omega-(1i/2)*Gamma;
[dipolemagarr,Energy]=eig(H);
dipolemag=dipolemagarr(:,modofsystem);
nofdmag=1;    
i=1;

positions=linspace(beg,endi,N);

for phi=0:0.01:2*pi
     for teta=0:0.01:pi
         for n=1:N
            d=positions(n);
            nofdmag=n;
            posS=[R*sin(teta)*cos(phi),R*sin(teta)*sin(phi),R*cos(teta)];
            absR=posS-[d,0,0];
            d1=sqrt(sum(absR.^2));
            absoluteR=absR/d1;
            %sinterm=norm(cross(absR,dvec))/d1;
            q0=abs(dipolemag(nofdmag));
            pha=angle(dipolemag(nofdmag));
            E1vec=cross(cross(dvec,absoluteR),absoluteR)*(1/d1)*q0*cos(-(w*d1)/c+pha);
            B1vec=cross(dvec,absoluteR)*(1/d1)*q0*cos(-(w*d1)/c+pha);
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
