# -*- coding: utf-8 -*-
"""
Created on Mon Feb  2 14:05:39 2026

@author: luisa
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust CSV-to-tracks analysis (fixed MSD, alpha fit, diffCoeff, drift-correct, interp)
"""

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tkinter as tk
from tkinter import filedialog

from scipy.interpolate import interp1d


# ----------------------------
# Core track import + metrics
# ----------------------------

def correctGap(x, frame0, frame):
    """Fill missing frames with NaN so each track has a contiguous frame axis."""
    filled = np.full_like(frame, np.nan, dtype=np.float64)
    idx = (np.array(frame0) - frame0[0]).astype(int)
    # guard against any weird indices
    idx = idx[(idx >= 0) & (idx < len(filled))]
    filled[idx] = x[:len(idx)]
    return np.array(filled)


def MSD2(x, y):
    """2D MSD for lag=1..n-1, robust to NaNs (gaps). Returns length n-1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n < 6:
        return np.full(max(n - 1, 1), np.nan)

    m = np.full(n - 1, np.nan)
    for lag in range(1, n):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        d2 = dx * dx + dy * dy
        if np.isfinite(d2).sum() >= 3:
            m[lag - 1] = np.nanmean(d2)
    return m


def MSD3(x, y, z):
    """3D MSD for lag=1..n-1, robust to NaNs (gaps). Returns length n-1."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(x)
    if n < 6:
        return np.full(max(n - 1, 1), np.nan)

    m = np.full(n - 1, np.nan)
    for lag in range(1, n):
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        dz = z[lag:] - z[:-lag]

        # Correct formula: dx^2 + dy^2 + dz^2
        d2 = dx * dx + dy * dy + dz * dz

        if np.isfinite(d2).sum() >= 3:
            m[lag - 1] = np.nanmean(d2)
    return m


def alphaCoeff(msd):
    """
    Fit log(MSD) ~ alpha * log(lag) + c using only positive finite points.
    Returns alpha or np.nan.
    """
    msd = np.asarray(msd, dtype=float)
    if msd.size < 3:
        return np.nan

    lag = np.arange(1, msd.size + 1, dtype=float)  # 1..len(msd)
    y = msd

    good = np.isfinite(lag) & np.isfinite(y) & (y > 0)
    lag = lag[good]
    y = y[good]

    if lag.size < 3:
        return np.nan

    lx = np.log(lag)
    ly = np.log(y)

    if np.std(lx) == 0 or np.std(ly) == 0:
        return np.nan

    slope, intercept = np.polyfit(lx, ly, 1)
    return slope


def diffCoeff(dt, msd):
    """
    Fit MSD ~ slope * t + intercept using first up to 5 valid MSD points.
    Note: msd[0] corresponds to lag=1 -> time=dt.
    Returns (slope, intercept) or (np.nan, np.nan).
    """
    msd = np.asarray(msd, dtype=float)

    good = np.isfinite(msd)
    idx = np.where(good)[0]
    if idx.size < 3:
        return np.nan, np.nan

    idx = idx[:5]
    t = (idx + 1) * dt
    y = msd[idx]

    slope, intercept = np.polyfit(t, y, 1)
    return slope, intercept


def gyradius(x, y, z):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    z = np.asarray(z, dtype=float)
    return np.sqrt(np.nanvar(x) + np.nanvar(y) + np.nanvar(z))


def extract_matrix(trk, field_name):
    """Pad track vectors to a matrix (numTracks x maxLen) with NaNs."""
    max_len = max(len(entry[field_name]) for entry in trk.values())
    v_matrix = np.array([
        np.pad(entry[field_name], (0, max_len - len(entry[field_name])), constant_values=np.nan)
        for entry in trk.values()
    ])
    return v_matrix


def import_tracks(name):
    df = pd.read_csv(name, skiprows=[1, 2, 3], encoding='ISO-8859-1')

    tr = np.unique(df.TRACK_ID)
    allx = df.POSITION_X.to_numpy()
    ally = df.POSITION_Y.to_numpy()
    allz = df.POSITION_Z.to_numpy()
    allt = df.POSITION_T.to_numpy()
    allr = df.RADIUS.to_numpy()
    allq = df.QUALITY.to_numpy()
    allf = df.FRAME.to_numpy()

    k = 0
    tracks = {}

    for i in tr:
        mask = (df.TRACK_ID == i)

        x = allx[mask]
        y = ally[mask]
        z = allz[mask]
        t = allt[mask]
        r = allr[mask]
        q = allq[mask]
        frame0 = allf[mask]  # count from 0!!!

        # sort by time
        a = np.argsort(t)
        t = t[a]
        x = x[a]
        y = y[a]
        z = z[a]
        r = r[a]
        q = q[a]
        frame0 = frame0[a]

        # robust dt
        if len(t) >= 2 and np.isfinite(t[0]) and np.isfinite(t[1]) and (t[1] != t[0]):
            dt = float(np.diff(t[0:2]).item())
        else:
            # fallback: infer from frames if possible, else set 1
            dt = 1.0

        frame = np.arange(int(np.nanmin(frame0)), int(np.nanmax(frame0)) + 1, 1)
        t = frame * dt

        x = correctGap(x, frame0, frame)
        y = correctGap(y, frame0, frame)
        z = correctGap(z, frame0, frame)
        r = correctGap(r, frame0, frame)
        q = correctGap(q, frame0, frame)

        xc = np.nanmean(x)
        yc = np.nanmean(y)
        zc = np.nanmean(z)
        rmean = np.nanmean(r)
        qmean = np.nanmean(q)
        ngap = int(np.isnan(x).sum())

        gyr = gyradius(x, y, z)

        # compute MSD/alpha/D robustly only if long enough
        if len(x) >= 6:
            msd2D = MSD2(x, y)
            msd3D = MSD3(x, y, z)
            alpha2 = alphaCoeff(msd2D)
            alpha3 = alphaCoeff(msd3D)

            slope2, intercept2 = diffCoeff(dt, msd2D)
            D2 = slope2 / 4 if np.isfinite(slope2) else np.nan

            slope3, intercept3 = diffCoeff(dt, msd3D)
            D3 = slope3 / 6 if np.isfinite(slope3) else np.nan
        else:
            msd2D = np.full(1, np.nan)
            msd3D = np.full(1, np.nan)
            alpha2 = np.nan
            alpha3 = np.nan
            D2 = np.nan
            D3 = np.nan
            intercept2 = np.nan
            intercept3 = np.nan

        if len(x) >= 5:
            tracks[k] = {
                "id": i,
                "x": x, "y": y, "z": z, "t": t, "frame": frame, "radius": r, "quality": q,
                "dt": dt, "nspots": len(x), "xc": xc, "yc": yc, "zc": zc, "rmean": rmean,
                "qmean": qmean, "ngap": ngap,
                "MSD2D": msd2D, "alpha2": alpha2, "MSD3D": msd3D, "alpha3": alpha3,
                "D2": D2, "D3": D3,
                # keeping your original keys even though these are intercepts
                "err2": intercept2, "err3": intercept3,
                "gyradius": gyr
            }
            k += 1

    return tracks


# ----------------------------
# Drift correction
# ----------------------------

def driftCorrectTrack(data, selMovie, doPlot):
    tracks = data[selMovie]["tracks"]

    tracksLen = np.array([tracks['nspots'] for tracks in tracks.values()])
    tracksChk = np.array([np.nanmean(tracks['x']) for tracks in tracks.values()])

    tracksSel = np.intersect1d(
        np.where(tracksLen == np.nanmax(tracksLen))[0],
        np.where(~np.isnan(tracksChk))[0]
    )

    # If selection is empty, fallback to all tracks with finite mean
    if tracksSel.size == 0:
        tracksSel = np.where(~np.isnan(tracksChk))[0]

    mx = np.vstack([tracks[key]["x"] for key in tracksSel])
    xdrift = np.nanmean(mx, 0)

    my = np.vstack([tracks[key]["y"] for key in tracksSel])
    ydrift = np.nanmean(my, 0)

    mz = np.vstack([tracks[key]["z"] for key in tracksSel])
    zdrift = np.nanmean(mz, 0)

    for key in tracks.keys():
        tracks[key]["xdrift"] = xdrift
        tracks[key]["ydrift"] = ydrift
        tracks[key]["zdrift"] = zdrift

        time = tracks[key]["frame"]
        time = time[time < len(xdrift)]  # trim if needed

        tracks[key]["xcorr"] = tracks[key]["x"][:len(time)] - xdrift[time] + xdrift[0]
        tracks[key]["ycorr"] = tracks[key]["y"][:len(time)] - ydrift[time] + ydrift[0]
        tracks[key]["zcorr"] = tracks[key]["z"][:len(time)] - zdrift[time] + zdrift[0]

        tracks[key]["XCcorr"] = np.nanmean(tracks[key]["xcorr"])
        tracks[key]["YCcorr"] = np.nanmean(tracks[key]["ycorr"])  # FIXED
        tracks[key]["ZCcorr"] = np.nanmean(tracks[key]["zcorr"])

        # MSD on corrected coords
        tracks[key]["MSD3corr"] = MSD3(tracks[key]["xcorr"], tracks[key]["ycorr"], tracks[key]["zcorr"])
        tracks[key]["MSD2corr"] = MSD2(tracks[key]["xcorr"], tracks[key]["ycorr"])

        # alpha on corrected MSD
        tracks[key]["alpha2corr"] = alphaCoeff(tracks[key]["MSD2corr"])
        tracks[key]["alpha3corr"] = alphaCoeff(tracks[key]["MSD3corr"])

        # diffusion on corrected MSD
        slope2, _ = diffCoeff(data[selMovie]["dt"], tracks[key]["MSD2corr"])
        tracks[key]["D2corr"] = slope2 / 4 if np.isfinite(slope2) else np.nan

        slope3, _ = diffCoeff(data[selMovie]["dt"], tracks[key]["MSD3corr"])
       # tracks[key]["D3corr"] = slope3 / 6 if np.isfinite(slope3) else np.nan  # FIXED
        D3corr = slope3 / 6 if np.isfinite(slope3) else np.nan
        tracks[key]["D3corr"] = D3corr if (np.isfinite(D3corr) and D3corr >= 0) else np.nan

        print(key)

    if doPlot:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(xdrift, ydrift, zdrift, label='drift')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        plt.title('data ' + str(selMovie))
        plt.show()

    return tracks, xdrift, ydrift, zdrift


# ----------------------------
# MSD interpolation (movie-level)
# ----------------------------

def MSDinterp(data):
    for movie in data:
        msdMean = np.nanmean(movie["MSD3Dcmat"], axis=0)

        dt = movie["dt"]
        t_ori = (np.arange(len(msdMean)) + 1) * dt  # lag 1.. -> time dt..

        good = np.isfinite(msdMean) & np.isfinite(t_ori)
        if good.sum() < 3:
            movie["timeOri"] = t_ori
            movie["MSD3MeanOri"] = msdMean
            movie["timeInt"] = np.array([])
            movie["MSD3MeanInt"] = np.array([])
            continue

        t_int = np.arange(t_ori[good].min(), t_ori[good].max(), dt)

        interpolator = interp1d(
            t_ori[good],
            msdMean[good],
            kind='quadratic',
            bounds_error=False,
            fill_value="extrapolate"
        )
        msd_int = interpolator(t_int)

        movie["timeOri"] = t_ori
        movie["MSD3MeanOri"] = msdMean
        movie["timeInt"] = t_int
        movie["MSD3MeanInt"] = msd_int

    return data


# ----------------------------
# MAIN
# ----------------------------

if __name__ == "__main__":
    # Folder selection dialog
    root = tk.Tk()
    root.withdraw()
    directory_path = filedialog.askdirectory()

    if not directory_path:
        raise SystemExit("No folder selected. Exiting.")

    data = []
    k = 0

    for filename in os.listdir(directory_path):
        path_im = os.path.join(directory_path, filename)
        if os.path.isfile(path_im) and filename.endswith('csv'):
            print(filename)
            tracks = import_tracks(path_im)

            if len(tracks) == 0:
                print("  -> no valid tracks found, skipping")
                continue

            dt = tracks[0]["dt"]
            MSD2Dmat = extract_matrix(tracks, "MSD2D")
            MSD3Dmat = extract_matrix(tracks, "MSD3D")

            data.append({
                "name": filename,
                "tracks": tracks,
                "dt": dt,
                "numTracks": len(tracks),
                "MSD2Dmat": MSD2Dmat,
                "MSD3Dmat": MSD3Dmat,
                "Condition": "TB"
            })
            k += 1

    print("done importing")

    # Drift correct + derived matrices
    for i, movie in enumerate(data):
        movie["tracks"], xdrift, ydrift, zdrift = driftCorrectTrack(data, i, 0)
        tracks = movie["tracks"]

        movie["xdrift"] = xdrift
        movie["ydrift"] = ydrift
        movie["zdrift"] = zdrift

        movie["XCcorr"] = np.array([trk["XCcorr"] for trk in tracks.values()])
        movie["YCcorr"] = np.array([trk["YCcorr"] for trk in tracks.values()])
        movie["ZCcorr"] = np.array([trk["ZCcorr"] for trk in tracks.values()])

        movie["MSD2Dcmat"] = extract_matrix(tracks, "MSD2corr")
        movie["MSD3Dcmat"] = extract_matrix(tracks, "MSD3corr")

        movie["D3corr"] = np.array([trk["D3corr"] for trk in tracks.values()])
        movie["alpha3corr"] = np.array([trk["alpha3corr"] for trk in tracks.values()])

    data = MSDinterp(data)

    # Save pkl next to the folder, using parent folder name as condition
    path = Path(directory_path)
    cond = path.parent.name

    out_path = Path(directory_path) / f"{cond}.pkl"
    with open(out_path, 'wb') as file:
        pickle.dump(data, file)

    print(f"Saved: {out_path}")

