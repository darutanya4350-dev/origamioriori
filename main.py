#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d

# origami_engine.py が同じディレクトリにあると仮定
from origami_engine import OrigamiModel


class OrigamiViewer:
    def __init__(self):
        self.model = OrigamiModel()
        self.model.set_initial_square()

        self.fig = plt.figure(figsize=(10, 8))
        plt.subplots_adjust(bottom=0.2)
        self.ax = self.fig.add_subplot(111, projection="3d")

        # スナップ候補
        self.snap_points = []
        # ユーザーが選択した 2 点（折り線定義用）
        self.selected_points = []
        
        # 選択状態（ノードベース）
        self.picked_node_id = None
        self.picked_point = None  # つまんだ位置（移動側の判定に使用）

        # イベント接続
        self.cid_click = self.fig.canvas.mpl_connect(
            "button_press_event", self.on_click
        )

        # ボタン配置
        ax_reset = plt.axes([0.15, 0.05, 0.1, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_reset.on_clicked(self.on_reset)

        # Undo は新エンジンで未実装のため省略
        
        ax_fold180 = plt.axes([0.59, 0.05, 0.15, 0.075])
        self.btn_fold180 = Button(ax_fold180, 'Fold 180°')
        self.btn_fold180.on_clicked(self.on_fold_180)
        
        ax_fold170 = plt.axes([0.75, 0.05, 0.15, 0.075])
        self.btn_fold170 = Button(ax_fold170, 'Bend 170°')
        self.btn_fold170.on_clicked(self.on_bend_170)

        self.redraw()

    # ---------- スナップ候補の更新 ----------

    def update_snap_points(self):
        self.snap_points = []
        # 全ノードのポリゴンを走査
        for nid, node in self.model.nodes.items():
            for poly in node.polygons:
                n = len(poly)
                # 頂点
                for v in poly:
                    self.snap_points.append({
                        "type": "vertex",
                        "coord3d": v.copy(),
                        "coord2d": None,
                        "node_id": nid
                    })
                # 辺の中点
                for i in range(n):
                    v0 = poly[i]
                    v1 = poly[(i + 1) % n]
                    mid = 0.5 * (v0 + v1)
                    self.snap_points.append({
                        "type": "midpoint",
                        "coord3d": mid,
                        "coord2d": None,
                        "node_id": nid
                    })
                # 面の重心（ノード選択用）
                center = np.mean(poly, axis=0)
                self.snap_points.append({
                    "type": "face",
                    "coord3d": center,
                    "coord2d": None,
                    "node_id": nid,
                    "normal": node.normal
                })

        self.update_projection2d()

    def update_projection2d(self):
        if not self.snap_points:
            return
        # 3D -> 2D 投影
        for p in self.snap_points:
            x, y, z = p["coord3d"]
            x2, y2, z2 = proj3d.proj_transform(x, y, z, self.ax.get_proj())
            p["coord2d"] = (x2, y2)
            p["depth"] = z2

    # ---------- クリックイベント ----------

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if event.button != 1:
            return

        # 既に2点選択済みの場合はリセット
        if len(self.selected_points) >= 2:
            self.selected_points = []
            self.redraw()

        self.update_projection2d()

        click_xy = np.array([event.x, event.y])
        search_radius = 20.0
        candidates = []

        for p in self.snap_points:
            if p["coord2d"] is None:
                continue
            # 画面座標へ変換
            xy_disp = self.ax.transData.transform(p["coord2d"])
            dist = np.linalg.norm(click_xy - xy_disp)
            if dist < search_radius:
                candidates.append(p)

        if not candidates:
            return

        # 最も手前（depth大）を選択
        nearest = max(candidates, key=lambda item: item["depth"])
        snapped_3d = nearest["coord3d"]
        print(f"Snapped to: {nearest['type']} at {snapped_3d}")

        # Shiftキー: ノード（面）を選択
        if event.key == "shift":
            self.picked_point = snapped_3d
            self.picked_node_id = nearest["node_id"]
            print(f"Picked Node ID: {self.picked_node_id}")
            self.redraw()
            return

        # 通常クリック: 点を選択
        self.selected_points.append(snapped_3d)
        self.ax.scatter(
            [snapped_3d[0]], [snapped_3d[1]], [snapped_3d[2]],
            color="red", s=30
        )
        self.fig.canvas.draw_idle()

    # ---------- ボタンイベント ----------

    def on_reset(self, event):
        self.model = OrigamiModel()
        self.model.set_initial_square()
        self.selected_points = []
        self.picked_node_id = None
        self.picked_point = None
        self.redraw()

    def on_fold_180(self, event):
        if len(self.selected_points) == 2:
            self.do_fold(self.selected_points[0], self.selected_points[1], 180.0, is_bend=False)
            self.selected_points = []
        else:
            print("Fold ignored: Please select 2 points.")

    def on_bend_170(self, event):
        if len(self.selected_points) == 2:
            self.do_fold(self.selected_points[0], self.selected_points[1], 170.0, is_bend=True)
            self.selected_points = []
        else:
            print("Bend ignored: Please select 2 points.")

    # ---------- 折り処理 ----------

    def do_fold(self, p1, p2, angle_deg, is_bend=False):
        # 操作対象ノードの決定
        target_node_id = self.picked_node_id
        if target_node_id is None:
            # ノードが1つしかなければそれを対象にする
            if len(self.model.nodes) == 1:
                target_node_id = list(self.model.nodes.keys())[0]
            else:
                print("Error: No node selected. Please Shift+Click a face to select the node to fold.")
                return

        # 垂直二等分線による折り軸の計算
        node = self.model.nodes[target_node_id]
        normal = node.normal
        
        mid = 0.5 * (p1 + p2)
        d = p2 - p1
        # 折り線方向 = normal x (p2 - p1)
        axis_dir = np.cross(normal, d)
        norm = np.linalg.norm(axis_dir)
        if norm < 1e-9:
            print("Invalid fold axis.")
            return
        axis_dir /= norm

        # 十分長い折り線を定義
        L = 10.0
        axis_start = mid - axis_dir * L
        axis_end = mid + axis_dir * L

        # 移動側（picked_pointがある側）が回転するように軸の向きを調整
        # エンジンは「負側」を回転させる仕様
        if self.picked_point is not None:
            # エンジンと同様の平面法線計算で符号を確認
            # ref = (0,0,1) or (1,0,0)
            ref = np.array([0.0, 0.0, 1.0])
            if np.all(np.isclose(axis_dir, ref)) or np.all(np.isclose(axis_dir, -ref)):
                ref = np.array([1.0, 0.0, 0.0])
            plane_n = np.cross(axis_dir, ref)
            plane_n /= np.linalg.norm(plane_n)
            
            dist = np.dot(self.picked_point - axis_start, plane_n)
            
            # picked_point が正側にある場合、軸を反転させて負側にする
            if dist > 0:
                axis_start, axis_end = axis_end, axis_start

        # 実行
        if is_bend:
            # 170度折り（新ノード生成）
            self.model.apply_soft_fold_to_node(target_node_id, axis_start, axis_end, angle_deg)
        else:
            # 180度折り（ノード内変形）
            angle_rad = np.deg2rad(angle_deg)
            self.model.apply_full_fold_to_node(target_node_id, axis_start, axis_end, angle_rad)

        # 選択解除
        self.picked_point = None
        self.picked_node_id = None
        self.redraw()

    # ---------- 描画 ----------

    def redraw(self):
        self.ax.clear()

        polys_to_draw = []
        face_colors = []

        # 全ノードを描画
        for nid, node in self.model.nodes.items():
            for poly in node.polygons:
                polys_to_draw.append(poly)
                # 選択中のノードは緑色
                if nid == self.picked_node_id:
                    face_colors.append("lightgreen")
                else:
                    face_colors.append("white")

        if polys_to_draw:
            collection = Poly3DCollection(
                polys_to_draw,
                facecolors=face_colors,
                edgecolors="k",
                linewidths=1.0,
                alpha=0.8,
            )
            self.ax.add_collection3d(collection)

        # 選択点（赤丸）
        if self.selected_points:
            xs = [p[0] for p in self.selected_points]
            ys = [p[1] for p in self.selected_points]
            zs = [p[2] for p in self.selected_points]
            self.ax.scatter(xs, ys, zs, color="red", s=30, label="Selected")

        # 軸設定
        self.ax.set_xlim(-0.5, 1.5)
        self.ax.set_ylim(-0.5, 1.5)
        self.ax.set_zlim(-0.5, 1.5)
        self.ax.set_xlabel("X")
        self.ax.set_ylabel("Y")
        self.ax.set_zlabel("Z")
        self.ax.set_title("Origami Viewer (Tree Node Version)")

        self.update_snap_points()
        self.fig.canvas.draw_idle()


def main():
    viewer = OrigamiViewer()
    plt.show()


if __name__ == "__main__":
    main()
