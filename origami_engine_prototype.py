#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class OrigamiModel:
    """
    折り紙の幾何状態（ポリゴン群）を管理し，
    折り操作（fold）を適用するエンジンクラス。
    """

    def __init__(self):
        # ポリゴンのリスト（各要素は shape=(N,3), dtype=float64）
        self.polygons: list[np.ndarray] = []
        self.history: list[list[np.ndarray]] = []

    def set_initial_square(self):
        """
        初期状態として 1x1 の正方形を z=0 平面上に生成する。
        """
        square = np.array(
            [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 0.0],
            ],
            dtype=np.float64,
        )
        self.polygons = [square]
        self.history = []

    # ========= 基本幾何ユーティリティ =========

    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(v)
        if np.isclose(norm, 0.0):
            return v
        return v / norm

    @staticmethod
    def _rotation_matrix_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        """
        任意軸 (単位ベクトル axis) と回転角 angle [rad] から
        3x3 の回転行列を生成（Rodrigues の回転公式）。[web:21][web:24]
        """
        axis = OrigamiModel._normalize(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c

        R = np.array(
            [
                [c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                [z * x * C - y * s, z * y * C + x * s, c + z * z * C],
            ],
            dtype=np.float64,
        )
        return R

    @staticmethod
    def _signed_distance_to_fold_plane(points, axis_start, axis_end):
        """
        折り線を含む平面に対する符号付き距離を計算する。
        折り線: axis_start -> axis_end
        平面法線 n = (axis_end - axis_start) × z軸 を仮定（簡易）。[web:21]
        """
        p0 = axis_start
        axis_dir = OrigamiModel._normalize(axis_end - axis_start)

        # z軸と平行な折り線を避けるため，適当に別のベクトルを選ぶ
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.all(np.isclose(axis_dir, ref)) or np.all(np.isclose(axis_dir, -ref)):
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)

        n = np.cross(axis_dir, ref)
        n = OrigamiModel._normalize(n)

        # 各点から平面までの符号付き距離
        # d = (p - p0)・n
        vec = points - p0
        d = np.dot(vec, n)
        return d, n

    # ========= 厚みの管理 =========

    def _apply_thickness_offset(self):
        """
        z 方向の微小オフセットによって，重なりを解消する簡易処理。
        現在は「ポリゴンのインデックスに応じて」順番に +ε を足すだけの実装。
        将来的に重なり判定をきちんと実装して差し替え可能。 [web:29]
        """
        epsilon = 1e-4  # 必要に応じて調整
        for i, poly in enumerate(self.polygons):
            self.polygons[i] = poly.copy()
            self.polygons[i][:, 2] += epsilon * i

    # ========= 履歴管理 (Undo) =========

    def undo(self):
        if self.history:
            self.polygons = self.history.pop()

    def _save_state(self):
        # 現在のポリゴンリストを複製して履歴に積む
        self.history.append([p.copy() for p in self.polygons])

    # ========= 折り操作のメイン =========

    def apply_fold(self, axis_start, axis_end, angle, apply_thickness=True, active_point=None):
        """
        折り線(axis_start -> axis_end)を回転軸とする折り操作を適用する。
        1. 各ポリゴンの各辺と「折り線を含む無限直線」の交点を求める
        2. 線をまたぐポリゴンを交点で2つに分割
        3. 指定側（平面の片側）のポリゴン群を回転
        4. 重なりに応じて z オフセットを加算
        """
        # 操作前に履歴保存
        self._save_state()

        axis_start = np.array(axis_start, dtype=np.float64)
        axis_end = np.array(axis_end, dtype=np.float64)
        axis_dir = self._normalize(axis_end - axis_start)

        # -----------------------------
        # 1. 折り線の「向き付き平面」と点の位置関係
        # -----------------------------
        def signed_distance(points):
            # ref ベクトルと折り線方向から平面法線を作る
            ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
            if np.all(np.isclose(axis_dir, ref)) or np.all(np.isclose(axis_dir, -ref)):
                ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
            n = np.cross(axis_dir, ref)
            n = self._normalize(n)
            vec = points - axis_start
            d = vec @ n
            return d, n

        # -----------------------------
        # 2. 辺と折り線の交点（線分と無限直線の交差）
        #    ・まず「折り線を含む平面」で頂点を +/0/- に分類
        #    ・符号が変わる辺上で平面との交点をとり、それを折り線上に射影
        # -----------------------------
        eps = 1e-9

        def segment_plane_intersection(p0, p1, plane_point, plane_normal):
            # p(t) = p0 + t (p1 - p0),  plane: (x - plane_point)·n = 0
            u = p1 - p0
            denom = np.dot(u, plane_normal)
            if np.isclose(denom, 0.0):
                return None
            t = np.dot(plane_point - p0, plane_normal) / denom
            if t < -eps or t > 1.0 + eps:
                return None
            return p0 + t * u

        def project_to_fold_line(q):
            # 点 q を折り線無限直線上に正射影
            v = q - axis_start
            t = np.dot(v, axis_dir)
            return axis_start + t * axis_dir

        # ポリゴンを「折り線で2つに切る」関数
        def split_polygon_by_line(poly):
            d, plane_normal = signed_distance(poly)

            # 完全に片側 or 線上なら分割不要
            if np.all(d >= -eps) or np.all(d <= eps):
                return [poly], []  # [positive側 or on-line], [negative側]
            if np.all(d <= eps) or np.all(d >= -eps):
                return [], [poly]

            # 頂点を順に見ながら、新しいポリゴンを構築
            pos_vertices = []
            neg_vertices = []

            n_vert = len(poly)
            for i in range(n_vert):
                p_curr = poly[i]
                p_next = poly[(i + 1) % n_vert]
                d_curr = d[i]
                d_next = d[(i + 1) % n_vert]

                def is_pos(val):
                    return val > eps

                def is_neg(val):
                    return val < -eps

                # 現在頂点を対応側に追加
                if is_pos(d_curr) or np.isclose(d_curr, 0.0, atol=eps):
                    pos_vertices.append(p_curr)
                if is_neg(d_curr) or np.isclose(d_curr, 0.0, atol=eps):
                    neg_vertices.append(p_curr)

                # 辺が平面を横切る場合は交点を計算して両側に入れる
                if (is_pos(d_curr) and is_neg(d_next)) or (is_neg(d_curr) and is_pos(d_next)):
                    inter = segment_plane_intersection(p_curr, p_next, axis_start, plane_normal)
                    if inter is not None:
                        # 交点を折り線上に射影して「線上」に載せる
                        inter_on_line = project_to_fold_line(inter)
                        pos_vertices.append(inter_on_line)
                        neg_vertices.append(inter_on_line)

            new_polys_pos = []
            new_polys_neg = []
            if len(pos_vertices) >= 3:
                new_polys_pos.append(np.array(pos_vertices, dtype=np.float64))
            if len(neg_vertices) >= 3:
                new_polys_neg.append(np.array(neg_vertices, dtype=np.float64))
            return new_polys_pos, new_polys_neg

        # -----------------------------
        # 3. 全ポリゴンに対し分割 → どちら側か分類
        # -----------------------------
        positive_side = []  # n·(p - axis_start) > 0 の側
        negative_side = []  # < 0 の側
        on_line = []        # ほぼ線上

        for poly in self.polygons:
            d, plane_normal = signed_distance(poly)

            if np.all(np.isclose(d, 0.0, atol=eps)):
                on_line.append(poly)
                continue

            # 分割が必要な場合
            if np.any(d > eps) and np.any(d < -eps):
                pos_polys, neg_polys = split_polygon_by_line(poly)
                positive_side.extend(pos_polys)
                negative_side.extend(neg_polys)
            else:
                # 片側だけの場合
                if np.all(d >= -eps):
                    positive_side.append(poly)
                elif np.all(d <= eps):
                    negative_side.append(poly)

        # -----------------------------
        # 4. 回転（Rodrigues の回転公式）
        # -----------------------------
        R = self._rotation_matrix_axis_angle(axis_dir, angle)

        def rotate_points(points):
            moved = points - axis_start
            rotated = moved @ R.T
            return rotated + axis_start

        # どちら側を回転させるか決定
        # デフォルトは negative_side を回転（positive_side を固定）
        rotate_positive = False

        if active_point is not None:
            # active_point がどちら側にあるか判定
            d_active, _ = signed_distance(active_point)
            if d_active > eps:
                rotate_positive = True
            elif d_active < -eps:
                rotate_positive = False

        if rotate_positive:
            rotated_side = [rotate_points(p) for p in positive_side]
            static_side = negative_side
        else:
            rotated_side = [rotate_points(p) for p in negative_side]
            static_side = positive_side

        # -----------------------------
        # 5. 新しいポリゴンリストを構成
        # -----------------------------
        self.polygons = []
        self.polygons.extend(static_side)
        self.polygons.extend(on_line)
        self.polygons.extend(rotated_side)

        # -----------------------------
        # 6. 厚み（z オフセット）の累積
        #    ここでは「同じ (x,y) を持つポリゴン頂点がどれだけ重なるか」を
        #    ざっくりカウントし、重なりレベルに応じて z に ε を足す。
        # -----------------------------
        if apply_thickness:
            self._apply_overlap_thickness()

        return self.get_polygons()

    def fold_by_perpendicular_bisector(self, p1, p2, angle, apply_thickness=True, active_point=None, face_normal=None):
        """
        2点 p1, p2 を重ねるような折り（垂直二等分線での折り）を適用する。
        """
        p1 = np.array(p1, dtype=np.float64)
        p2 = np.array(p2, dtype=np.float64)

        # 中点
        mid = 0.5 * (p1 + p2)
        # 紙の法線（指定がなければ z 軸と仮定）
        if face_normal is not None:
            n = self._normalize(np.array(face_normal, dtype=np.float64))
        else:
            n = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        # 線分方向
        d = p2 - p1

        # 折り線方向 = n × d （紙面内で P1P2 に垂直）
        axis_dir = np.cross(n, d)
        norm = np.linalg.norm(axis_dir)

        if np.isclose(norm, 0.0):
            return self.get_polygons()

        axis_dir /= norm
        L = 10.0  # 折り線を十分長く取る
        axis_start = mid - axis_dir * L
        axis_end = mid + axis_dir * L

        return self.apply_fold(axis_start, axis_end, angle, apply_thickness=apply_thickness, active_point=active_point)

    def _apply_overlap_thickness(self):
        """
        ポリゴン間の重なりを簡易に判定し，重なりレベルに応じて
        z に微小オフセットを加える。

        アルゴリズム（簡略版）:
        - 各ポリゴン頂点の (x,y) を丸めてキーとし，何枚のポリゴンが
          その近傍を共有しているかをカウント
        - 同じキーを持つポリゴンに対して，インデックス順に ε を加算
        """
        epsilon = 5e-5
        # (key -> list of (poly_index, vertex_index))
        grid = {}

        # まず座標を走査してグリッド化
        for pi, poly in enumerate(self.polygons):
            for vi, v in enumerate(poly):
                # 数値誤差を吸収するために少し丸めたキーを使う
                key = (round(v[0], 6), round(v[1], 6))
                grid.setdefault(key, []).append((pi, vi))

        # 各キーごとに，重なり枚数に応じて z オフセットを付加
        for key, entries in grid.items():
            # エントリを poly_index でソートして層を決める
            entries_sorted = sorted(entries, key=lambda t: t[0])
            for layer, (pi, vi) in enumerate(entries_sorted):
                self.polygons[pi][vi, 2] += epsilon * layer

    def get_polygons(self):
        """
        現在のポリゴンリストを返す。
        """
        return self.polygons
