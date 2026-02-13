#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from origami_engine_prototype import OrigamiModel
from mpl_toolkits.mplot3d import proj3d


class OrigamiViewer:
    def __init__(self):
        self.model = OrigamiModel()
        self.model.set_initial_square()

        self.fig = plt.figure()
        plt.subplots_adjust(bottom=0.2)  # ボタン用のスペースを確保
        self.ax = self.fig.add_subplot(111, projection="3d")

        # スナップ候補（3D座標と投影2D座標を持つ）
        self.snap_points = []  # list[dict]
        # ユーザーが選択した 2 点（3D 座標）
        self.selected_points = []
        # ユーザーがつまんだ点（折り操作で動かす側の基準点）
        self.picked_point = None
        self.picked_normal = None  # つまんだ面の法線
        self.picked_face_index = None  # つまんだ面のインデックス

        # イベント接続
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

        # ボタンの配置
        ax_reset = plt.axes([0.15, 0.05, 0.1, 0.075])
        ax_undo = plt.axes([0.26, 0.05, 0.1, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.on_reset)
        self.btn_undo = Button(ax_undo, 'Undo')
        self.btn_undo.on_clicked(self.on_undo)

        ax_fold180 = plt.axes([0.59, 0.05, 0.15, 0.075])
        ax_fold170 = plt.axes([0.75, 0.05, 0.15, 0.075])
        self.btn_fold180 = Button(ax_fold180, 'Fold 180°')
        self.btn_fold180.on_clicked(self.on_fold_180)
        self.btn_fold170 = Button(ax_fold170, 'Bend 170°')
        self.btn_fold170.on_clicked(self.on_bend_170)

        self.redraw()

    # ---------- スナップ候補の更新 ----------

    def update_snap_points(self, polygons):
        self.snap_points = []
        for poly_idx, poly in enumerate(polygons):
            n = len(poly)
            # 頂点
            for v in poly:
                self.snap_points.append(
                    {
                        "type": "vertex",
                        "coord3d": v.copy(),
                        "coord2d": None,  # 後で投影する
                        "poly_index": poly_idx,
                    }
                )
            # 辺の中点
            for i in range(n):
                v0 = poly[i]
                v1 = poly[(i + 1) % n]
                mid = 0.5 * (v0 + v1)
                self.snap_points.append(
                    {
                        "type": "midpoint",
                        "coord3d": mid,
                        "coord2d": None,
                        "poly_index": poly_idx,
                    }
                )
            # 面の重心（つまむ用）
            center = np.mean(poly, axis=0)
            # 法線計算
            normal = np.array([0.0, 0.0, 1.0])
            if len(poly) >= 3:
                v0, v1, v2 = poly[0], poly[1], poly[2]
                vec1 = v1 - v0
                vec2 = v2 - v1
                cross = np.cross(vec1, vec2)
                if np.linalg.norm(cross) > 1e-9:
                    normal = cross / np.linalg.norm(cross)
            self.snap_points.append(
                {
                    "type": "face",
                    "coord3d": center,
                    "coord2d": None,
                    "normal": normal,
                    "poly_index": poly_idx,
                }
            )

        # 投影を更新
        self.update_projection2d()

    def update_projection2d(self):
        # 3D -> 2D 投影
        for p in self.snap_points:
            x, y, z = p["coord3d"]
            x2, y2, z2 = proj3d.proj_transform(x, y, z, self.ax.get_proj())
            p["coord2d"] = (x2, y2)
            p["depth"] = z2  # 深度情報を保存（大きいほど手前）

    # ---------- クリックイベント ----------

    def on_click(self, event):
        # 軸の外や右クリックなどは無視
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return
        if event.xdata is None or event.ydata is None:
            return

        # 既に2点選択済みの状態でクリックされたら、選択をリセットしてやり直す
        if len(self.selected_points) >= 2:
            self.selected_points = []
            self.redraw()  # 赤点を消すために再描画

        # 最新の投影を更新（視点を動かしたあとに対応させるため）
        self.update_projection2d()

        # クリック位置（図全体の座標系）に最も近い snap point を探す
        click_xy = np.array([event.x, event.y], dtype=np.float64)

        # クリック位置に近い候補（ピクセル距離）を探し、その中で最も手前（depth大）を選ぶ
        search_radius = 20.0  # ピクセル
        candidates = []

        for p in self.snap_points:
            x2, y2 = self.ax.transData.transform(p["coord2d"])
            pt_xy = np.array([x2, y2], dtype=np.float64)
            d = np.linalg.norm(click_xy - pt_xy)
            if d < search_radius:
                candidates.append(p)

        if not candidates:
            return

        # depth が大きいほうが手前（視点に近い）
        nearest = max(candidates, key=lambda item: item["depth"])

        snapped_3d = nearest["coord3d"]
        print("Snapped to:", nearest["type"], snapped_3d)

        # Shiftキーが押されていたら「つまむ」操作
        if event.key == "shift":
            self.picked_point = snapped_3d
            # 面を選択した場合は法線も保存する
            if nearest["type"] == "face":
                self.picked_normal = nearest.get("normal")
                self.picked_face_index = nearest.get("poly_index")
            else:
                self.picked_normal = None
                self.picked_face_index = None
            self.redraw()
            return

        self.selected_points.append(snapped_3d)

        # プロット上でも選択点を強調表示
        self.ax.scatter(
            [snapped_3d[0]],
            [snapped_3d[1]],
            [snapped_3d[2]],
            color="red",
            s=30,
        )
        self.fig.canvas.draw_idle()

    # ---------- ボタンイベント ----------

    def on_reset(self, event):
        self.model.set_initial_square()
        self.selected_points = []
        self.picked_point = None
        self.picked_normal = None
        self.picked_face_index = None
        self.redraw()

    def on_undo(self, event):
        self.model.undo()
        self.selected_points = []
        self.picked_point = None
        self.picked_normal = None
        self.picked_face_index = None
        self.redraw()

    def on_fold_180(self, event):
        if len(self.selected_points) == 2:
            self.fold_by_perpendicular_bisector(self.selected_points[0], self.selected_points[1], 180.0)
            self.selected_points = []
        else:
            print(f"Fold ignored: {len(self.selected_points)} points selected. Please select exactly 2 points for the fold line.")

    def on_bend_170(self, event):
        if len(self.selected_points) == 2:
            # 曲げ（Bend）の場合は厚み処理（重なり補正）を行わない
            self.fold_by_perpendicular_bisector(self.selected_points[0], self.selected_points[1], 160.0, apply_thickness=False)
            self.selected_points = []
        else:
            print(f"Bend ignored: {len(self.selected_points)} points selected. Please select exactly 2 points for the fold line.")

    # ---------- 垂直二等分線による fold ----------

    def fold_by_perpendicular_bisector(self, p1, p2, angle_deg, apply_thickness=True):
        angle = np.deg2rad(angle_deg)
        polygons = self.model.fold_by_perpendicular_bisector(
            p1, p2, angle, 
            apply_thickness=apply_thickness,
            active_point=self.picked_point,
            face_normal=self.picked_normal
        )
        self.picked_point = None  # 操作後は選択解除
        self.picked_normal = None
        self.picked_face_index = None
        self.redraw(polygons)

    # ---------- 描画 ----------

    def redraw(self, polygons=None):
        if polygons is None:
            polygons = self.model.get_polygons()

        self.ax.clear()

        # 面ごとの色を設定（選択された面は緑、それ以外は白）
        face_colors = []
        for i in range(len(polygons)):
            if i == self.picked_face_index:
                face_colors.append("lightgreen")
            else:
                face_colors.append("white")

        poly3d = [poly.tolist() for poly in polygons]
        collection = Poly3DCollection(
            poly3d,
            facecolors=face_colors,
            edgecolors="k",
            linewidths=1.0,
            alpha=0.8,
        )
        self.ax.add_collection3d(collection)

        # 選択中の点があれば赤色で表示（再描画で消えないようにする）
        if self.selected_points:
            xs = [p[0] for p in self.selected_points]
            ys = [p[1] for p in self.selected_points]
            zs = [p[2] for p in self.selected_points]
            self.ax.scatter(xs, ys, zs, color="red", s=30, label="Selected")

        # 縮尺が変わらないように表示範囲を固定する
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.set_zlim(-0.5, 1.5)

        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("3D Origami Viewer with Snapping")

        # スナップ候補も更新
        self.update_snap_points(polygons)

        self.fig.canvas.draw_idle()


def main():
    viewer = OrigamiViewer()
    plt.show()


if __name__ == "__main__":
    main()
