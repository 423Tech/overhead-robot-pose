"""Field trajectory plot from a pose.csv.

    plot_traj.py POSECSV OUT.png [--half-long M] [--half-short M] [--every N]

Draws the path, a dot every N-th fix and an arrow where yaw is available.
A stationary cluster or a straight line across the field is an artefact of the
detector, not motion - this plot exists to spot exactly that.
"""
import argparse, csv, sys, math
import numpy as np, cv2

HL, HS = 1.095, 0.79


def plot(path, out, hl=HL, hs=HS, every=90):
    rows=list(csv.DictReader(open(path)))
    W,H=1000,760
    img=np.full((H,W,3),245,np.uint8)
    def px(x,y):
        return (int(W/2 + x/1.25*(W/2-40)), int(H/2 - y/0.95*(H/2-40)))
    # field
    cv2.rectangle(img,px(-hl,hs),px(hl,-hs),(200,200,200),2)
    cv2.circle(img,px(0,0),int(0.30/1.25*(W/2-40)),(210,210,210),2)
    cv2.line(img,px(0,hs),px(0,-hs),(225,225,225),1)
    pts=[]
    for r in rows:
        if r["x"]!="NaN":
            pts.append((float(r["x"]),float(r["y"]),float(r["timestamp"]),None if r["yaw"]=="NaN" else float(r["yaw"])))
    for i in range(1,len(pts)):
        t0=pts[i-1][2]; t1=pts[i][2]
        if t1-t0>0.5: continue
        c=(int(120+120*math.sin(t1*0.01)),80,200)
        cv2.line(img,px(pts[i-1][0],pts[i-1][1]),px(pts[i][0],pts[i][1]),c,1)
    n=0
    for (x,y,t,yaw) in pts[::max(1, every)]:
        p=px(x,y); cv2.circle(img,p,2,(0,0,255),-1)
        if yaw is not None:
            a=math.radians(yaw)
            q=(int(p[0]+28*math.cos(a)), int(p[1]-28*math.sin(a)))
            cv2.arrowedLine(img,p,q,(0,150,0),1,tipLength=0.3)
            n+=1
    cv2.putText(img,f"{path}: {len(pts)} fixes, arrows every {every}th frame",(10,24),cv2.FONT_HERSHEY_SIMPLEX,0.6,(30,30,30),1)
    cv2.putText(img,"+x -> yellow goal (right), +y = up",(10,48),cv2.FONT_HERSHEY_SIMPLEX,0.55,(90,90,90),1)
    cv2.imwrite(out,img)
    print(out,"fixes",len(pts),"yaw arrows",n)

if __name__=="__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("posecsv")
    ap.add_argument("out")
    ap.add_argument("--half-long", type=float, default=HL)
    ap.add_argument("--half-short", type=float, default=HS)
    ap.add_argument("--every", type=int, default=90)
    a = ap.parse_args()
    plot(a.posecsv, a.out, a.half_long, a.half_short, a.every)
