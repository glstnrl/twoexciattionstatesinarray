using QuantumOptics
using LinearAlgebra
using SparseArrays
using Plots

N=3
np=2
hbar=1.054571817e-34
eps=8.8541878128e-12
c=2.99792458e8
mu=2.54e-29
cosofangle=0.75

space1=NLevelBasis(N)
fock=bosonstates(space1,np)
onehot=ManyBodyBasis(space1,fock)
dimension=length(fock)
Id=identityoperator(onehot)

H=zeros(Float64,N,N)
r=zeros(Float64,N,N)
ege=100
Gamma=zeros(Float64,N,N)
Omega=zeros(Float64,N,N)
reeig1=zeros(Float64,ege)
reeig2=zeros(Float64,ege)
reeig3=zeros(Float64,ege)
imeig1=zeros(Float64,ege)
imeig2=zeros(Float64,ege)
imeig3=zeros(Float64,ege)
lvec=zeros(Float64,ege)

for l=1:ege
    global H
    lamda0=0.01*1e-6
    k0=2*pi/lamda0
    w=k0*c
    for i=1:N
        global H
        for j=1:N
            global H
            ri=0.01*lamda0*i*l
            rj=0.01*lamda0*j*l
            r[i,j]=abs(rj-ri)
            mudotr=cosofangle
            if i==j
            Gamma[i,j]=((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)
            end
            if i!=j  
            Omega[i,j]=(3/4)*((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)*(((-(1-mudotr^2)*cos(k0*r[i,j]))/(k0*r[i,j]))+((1-3*mudotr^2)*((sin(k0*r[i,j]))/(k0*r[i,j])^2+cos(k0*r[i,j])/(k0*r[i,j])^3)))
            Gamma[i,j]=(3/2)*((w^3)*(mu^2))/(3*pi*eps*hbar*(c)^3)*((1-mudotr^2)*(sin(k0*r[i,j])/(k0*r[i,j]))+(1-3*mudotr^2)*(cos(k0*r[i,j])/(k0*r[i,j])^2-sin(k0*r[i,j])/(k0*r[i,j])^3))
            end
        end
    end
    H=(-im/2)*Gamma+Omega
    Energy=eigvals(H)
    reeig1[l]=real(Energy[1])
    reeig2[l]=real(Energy[2])
    reeig3[l]=real(Energy[3])
    imeig1[l]=imag(Energy[1])
    imeig2[l]=imag(Energy[2])
    imeig3[l]=imag(Energy[3])
    lvec[l]=l
    println(Energy)
end

gr()

p1 = plot(lvec, [reeig1 reeig2 reeig3],
          label=["Re E1" "Re E2" "Re E3"],
          title="Real eigenvalues")

p2 = plot(lvec, [imeig1 imeig2 imeig3],
          label=["Im E1" "Im E2" "Im E3"],
          title="Imag eigenvalues")

p = plot(p1, p2, layout=(2,1), size=(800,600))

display(p)

readline()



