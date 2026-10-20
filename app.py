import io
import os
import tempfile
import zipfile

import cv2
import streamlit as st

import remover as R

st.set_page_config(page_title="Watermark Remover", page_icon="🎬", layout="wide")
st.title("🎬 Watermark Remover")
st.caption("Use only on videos/images you own or have permission to edit.")

tab_video, tab_image, tab_moving = st.tabs(["🎥 Video", "🖼️ Image", "🎯 Moving watermark"])


def rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------- VIDEO
with tab_video:
    up = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"], key="vid")
    if up:
        if st.session_state.get("fname") != up.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(up.name)[1])
            tmp.write(up.read())
            tmp.close()
            st.session_state.update(fname=up.name, src=tmp.name, out=None)

        src = st.session_state["src"]
        info = R.video_info(src)
        st.write(f"**{info['w']}×{info['h']}** · {info['fps']:.1f} fps · {info['frames']} frames")

        col_ctrl, col_prev = st.columns([1, 2])
        mask = None
        with col_ctrl:
            mode = st.radio("Watermark selection", ["Manual box", "Auto-detect (static overlay)"], key="vmode")
            method = st.selectbox("Inpaint method", ["telea", "ns"], key="vmethod")
            radius = st.slider("Inpaint radius", 1, 15, 4, key="vradius")
            frame_idx = st.slider("Preview frame", 0, max(info["frames"] - 1, 0), 0)
            frame = R.read_frame(src, frame_idx)

            if mode == "Manual box":
                x = st.slider("X", 0, info["w"] - 1, int(info["w"] * 0.75))
                y = st.slider("Y", 0, info["h"] - 1, int(info["h"] * 0.85))
                w = st.slider("Width", 1, info["w"] - x, min(int(info["w"] * 0.2), info["w"] - x))
                h = st.slider("Height", 1, info["h"] - y, min(int(info["h"] * 0.1), info["h"] - y))
                mask = R.rect_mask(frame.shape[:2], x, y, w, h)
            else:
                std_t = st.slider("Static sensitivity", 2.0, 30.0, 10.0)
                grow = st.slider("Mask grow (px)", 3, 31, 9, step=2)
                try:
                    mask = R.auto_mask(src, std_thresh=std_t, dilate=grow)
                except ValueError as e:
                    st.error(str(e))

        if mask is not None:
            with col_prev:
                st.image(rgb(R.overlay_mask(frame, mask)),
                         caption="Red = area to be removed", use_container_width=True)

            if st.button("Remove watermark", type="primary", key="vrun"):
                if mask.max() == 0:
                    st.error("No watermark area selected.")
                else:
                    bar = st.progress(0.0)
                    out = os.path.join(tempfile.gettempdir(), f"clean_{up.name.rsplit('.', 1)[0]}.mp4")
                    with st.spinner("Processing..."):
                        R.process_video(src, out, mask, method=method, radius=radius,
                                        progress=lambda p: bar.progress(p))
                    st.session_state["out"] = out

        if st.session_state.get("out"):
            st.success("Done!")
            st.video(st.session_state["out"])
            with open(st.session_state["out"], "rb") as f:
                st.download_button("Download cleaned video", f, file_name="cleaned.mp4", mime="video/mp4")

# ---------------------------------------------------------------- IMAGE
with tab_image:
    files = st.file_uploader("Upload image(s)", type=["png", "jpg", "jpeg", "webp", "bmp"],
                             accept_multiple_files=True, key="img")
    if files:
        try:
            images = [(f.name, R.decode_image(f.getvalue())) for f in files]
        except ValueError as e:
            st.error(str(e))
            st.stop()

        first_name, first = images[0]
        H, W = first.shape[:2]
        st.write(f"**{len(images)}** image(s) · first: {W}×{H}")

        col_ctrl, col_prev = st.columns([1, 2])
        with col_ctrl:
            modes = ["Manual box"]
            same_size = [im for _, im in images if im.shape[:2] == (H, W)]
            if len(same_size) >= 5:
                modes.append("Auto-detect (same watermark on 5+ images)")
            mode = st.radio("Watermark selection", modes, key="imode")
            method = st.selectbox("Inpaint method", ["telea", "ns"], key="imethod")
            radius = st.slider("Inpaint radius", 1, 15, 4, key="iradius")

            auto = None
            if mode == "Manual box":
                xp = st.slider("X %", 0.0, 100.0, 75.0, 0.5)
                yp = st.slider("Y %", 0.0, 100.0, 85.0, 0.5)
                wp = st.slider("Width %", 0.5, 100.0 - xp, min(20.0, 100.0 - xp), 0.5)
                hp = st.slider("Height %", 0.5, 100.0 - yp, min(10.0, 100.0 - yp), 0.5)
            else:
                std_t = st.slider("Static sensitivity", 2.0, 30.0, 10.0, key="istd")
                grow = st.slider("Mask grow (px)", 3, 31, 9, step=2, key="igrow")
                auto = R.auto_mask_from_frames(same_size, std_thresh=std_t, dilate=grow)

        def mask_for(img):
            if auto is not None:
                if auto.shape == img.shape[:2]:
                    return auto
                return cv2.resize(auto, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
            return R.pct_rect_mask(img.shape[:2], xp, yp, wp, hp)

        with col_prev:
            st.image(rgb(R.overlay_mask(first, mask_for(first))),
                     caption="Red = area to be removed", use_container_width=True)

        if st.button("Remove watermark", type="primary", key="irun"):
            results = []
            bar = st.progress(0.0)
            for i, (name, img) in enumerate(images):
                out = R.inpaint_image(img, mask_for(img), method=method, radius=radius)
                results.append((os.path.splitext(name)[0] + "_clean.png", out))
                bar.progress((i + 1) / len(images))
            st.session_state["img_results"] = results

        results = st.session_state.get("img_results")
        if results:
            st.success("Done!")
            if len(results) == 1:
                name, out = results[0]
                st.image(rgb(out), use_container_width=True)
                st.download_button("Download cleaned image", R.encode_png(out),
                                   file_name=name, mime="image/png")
            else:
                for name, out in results[:6]:
                    st.image(rgb(out), caption=name, width=300)
                buf = io.BytesIO()
                with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
                    for name, out in results:
                        z.writestr(name, R.encode_png(out))
                st.download_button("Download all (ZIP)", buf.getvalue(),
                                   file_name="cleaned_images.zip", mime="application/zip")


# ------------------------------------------------------- MOVING WATERMARK
with tab_moving:
    st.info("Draw a box around the watermark on a reference frame. It is then located in "
            "every frame, even when it drifts, bounces or jumps between positions.")
    upm = st.file_uploader("Upload video", type=["mp4", "mov", "avi", "mkv", "webm"], key="mvid")
    if upm:
        if st.session_state.get("mname") != upm.name:
            tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(upm.name)[1])
            tmp.write(upm.read())
            tmp.close()
            st.session_state.update(mname=upm.name, msrc=tmp.name, mout=None, mtest=None)

        msrc = st.session_state["msrc"]
        mi = R.video_info(msrc)
        W, H = mi["w"], mi["h"]
        st.write(f"**{W}×{H}** · {mi['fps']:.1f} fps · {mi['frames']} frames")

        c1, c2 = st.columns([1, 2])
        with c1:
            ref_idx = st.slider("Reference frame (watermark visible)", 0,
                                max(mi["frames"] - 1, 0), 0, key="mref")
            ref = R.read_frame(msrc, ref_idx)
            bx = st.slider("Box X", 0, W - 2, int(W * 0.4), key="mx")
            by = st.slider("Box Y", 0, H - 2, int(H * 0.45), key="my")
            bw = st.slider("Box width", 2, max(W - bx, 3), min(int(W * 0.2), max(W - bx, 3)), key="mw")
            bh = st.slider("Box height", 2, max(H - by, 3), min(int(H * 0.08), max(H - by, 3)), key="mh")
            bw, bh = min(bw, W - bx), min(bh, H - by)
            box = (bx, by, bw, bh)

            text_mask = st.checkbox("Remove only text pixels (cleaner result)", True)
            thr = st.slider("Match threshold", 0.10, 0.90, 0.35, 0.01,
                            help="Lower = detects fainter watermark, but may cause false hits")
            margin = st.slider("Local search margin (px)", 0, 300, 80)
            hold = st.slider("Keep removing for N frames if lost", 0, 30, 5)
            multi = st.checkbox("Handle size changes (multi-scale)", False)
            scales = (0.85, 1.0, 1.15) if multi else (1.0,)
            method = st.selectbox("Inpaint method", ["telea", "ns"], key="mmethod")
            radius = st.slider("Inpaint radius", 1, 15, 4, key="mradius")

        with c2:
            vis = ref.copy()
            cv2.rectangle(vis, (bx, by), (bx + bw, by + bh), (0, 255, 0), 3)
            st.image(rgb(vis), caption="Reference frame - box must tightly enclose the watermark",
                     use_container_width=True)
            _, tmask = R.build_template(ref, box, text_mask)
            crop = ref[by:by + bh, bx:bx + bw]
            st.image(rgb(R.overlay_mask(crop, tmask)),
                     caption="Red = pixels that will be removed", width=360)

        b1, b2 = st.columns(2)
        if b1.button("🔍 Test detection on sample frames"):
            with st.spinner("Testing..."):
                st.session_state["mtest"] = R.detect_preview(msrc, ref, box, thr, scales, text_mask)
        if b2.button("Remove moving watermark", type="primary", key="mrun"):
            bar = st.progress(0.0)
            out = os.path.join(tempfile.gettempdir(), f"clean_dyn_{upm.name.rsplit('.', 1)[0]}.mp4")
            with st.spinner("Tracking and removing..."):
                stats = R.process_video_dynamic(msrc, out, ref, box, method, radius, thr, scales,
                                                margin, hold, text_mask,
                                                progress=lambda p: bar.progress(p))
            st.session_state["mout"] = out
            st.session_state["mstats"] = stats

        if st.session_state.get("mtest"):
            st.caption("Green = detected (score >= threshold), red = missed. Adjust threshold/box if needed.")
            cols = st.columns(3)
            for i, (img, sc) in enumerate(st.session_state["mtest"]):
                cols[i % 3].image(rgb(img), caption=f"score {sc:.2f}", use_container_width=True)

        if st.session_state.get("mout"):
            stats = st.session_state.get("mstats", {})
            st.success(f"Done! Watermark found in {stats.get('detected', 0)} of {stats.get('frames', 0)} frames.")
            st.video(st.session_state["mout"])
            with open(st.session_state["mout"], "rb") as f:
                st.download_button("Download cleaned video", f, file_name="cleaned_dynamic.mp4",
                                   mime="video/mp4", key="mdl")