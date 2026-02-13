#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np


class SurfaceNode:
    """1つの“平面グループ”＝同じ法線を持つポリゴン集合を表すノード"""

    def __init__(self, node_id, polygons, normal, parent=None):
        # polygons: list[np.ndarray], 各 poly は (N,3)
        self.id = node_id
        self.polygons = polygons
        self.normal = self._normalize(normal)
        self.parent = parent
        self.children = []  # list[SurfaceNode]

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if np.isclose(n, 0.0):
            return v
        return v / n


class OrigamiModel:
    """
    木構造で折り状態を管理する新バージョン。
    rootノード：初期の紙
    170度折り：対象ノードから新しい子ノードを生やす
    180度折り：同一ノード内で完結（ノード分割しない）
    """

    def __init__(self):
        self.nodes = {}         # node_id -> SurfaceNode
        self.root_id = None
        self.next_node_id = 0

    def _new_node_id(self):
        nid = self.next_node_id
        self.next_node_id += 1
        return nid

    def set_initial_square(self):
        square = np.array(
            [[0.0, 0.0, 0.0],
             [1.0, 0.0, 0.0],
             [1.0, 1.0, 0.0],
             [0.0, 1.0, 0.0]],
            dtype=np.float64
        )
        normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        nid = self._new_node_id()
        node = SurfaceNode(nid, [square], normal, parent=None)
        self.nodes[nid] = node
        self.root_id = nid

    # ========== 基本アクセサ ==========

    def get_leaf_nodes(self):
        """現在の“端点”ノードを返す（子を持たないノード）"""
        leaves = []
        for node in self.nodes.values():
            if len(node.children) == 0:
                leaves.append(node)
        return leaves

    def get_all_polygons(self):
        """描画用：全ノードのポリゴンをひとまとめに返す"""
        polys = []
        for node in self.nodes.values():
            polys.extend(node.polygons)
        return [p.copy() for p in polys]

    # ========== 幾何ユーティリティ（既存コード流用） ==========

    @staticmethod
    def _normalize(v):
        v = np.asarray(v, dtype=np.float64)
        n = np.linalg.norm(v)
        if np.isclose(n, 0.0):
            return v
        return v / n

    @staticmethod
    def _rotation_matrix_axis_angle(axis, angle):
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

    def _signed_distance_to_plane(self, points, axis_start, axis_end):
        axis_dir = self._normalize(axis_end - axis_start)
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if np.all(np.isclose(axis_dir, ref)) or np.all(np.isclose(axis_dir, -ref)):
            ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
        n = np.cross(axis_dir, ref)
        n = self._normalize(n)
        vec = points - axis_start
        d = np.dot(vec, n)
        return d, n

    def _segment_plane_intersection(self, p0, p1, plane_point, plane_normal):
        eps = 1e-9
        u = p1 - p0
        denom = np.dot(u, plane_normal)
        if np.isclose(denom, 0.0):
            return None
        t = np.dot(plane_point - p0, plane_normal) / denom
        if t < -eps or t > 1.0 + eps:
            return None
        return p0 + t * u

    def _project_to_fold_line(self, q, axis_start, axis_dir):
        v = q - axis_start
        t = np.dot(v, axis_dir)
        return axis_start + t * axis_dir

    def _split_polygon_by_line(self, poly, axis_start, axis_end):
        eps = 1e-9
        d, plane_normal = self._signed_distance_to_plane(poly, axis_start, axis_end)
        if np.all(d >= -eps) or np.all(d <= eps):
            return [poly], []
        if np.all(d <= eps) or np.all(d >= -eps):
            return [], [poly]

        pos_vertices = []
        neg_vertices = []
        n_vert = len(poly)
        axis_dir = self._normalize(axis_end - axis_start)

        for i in range(n_vert):
            p_curr = poly[i]
            p_next = poly[(i + 1) % n_vert]
            d_curr = d[i]
            d_next = d[(i + 1) % n_vert]

            if d_curr > eps or np.isclose(d_curr, 0.0, atol=eps):
                pos_vertices.append(p_curr)
            if d_curr < -eps or np.isclose(d_curr, 0.0, atol=eps):
                neg_vertices.append(p_curr)

            if (d_curr > eps and d_next < -eps) or (d_curr < -eps and d_next > eps):
                inter = self._segment_plane_intersection(p_curr, p_next, axis_start, plane_normal)
                if inter is not None:
                    inter_on_line = self._project_to_fold_line(inter, axis_start, axis_dir)
                    pos_vertices.append(inter_on_line)
                    neg_vertices.append(inter_on_line)

        new_pos = [np.array(pos_vertices, dtype=np.float64)] if len(pos_vertices) >= 3 else []
        new_neg = [np.array(neg_vertices, dtype=np.float64)] if len(neg_vertices) >= 3 else []
        return new_pos, new_neg

    # ========== 180度折り（ノード分割なし） ==========

    def apply_full_fold_to_node(self, node_id, axis_start, axis_end, angle=np.pi):
        """
        指定ノード内の「一方の側」を180度折り切る。
        ノードはそのままで、ポリゴンだけ更新（従来に近い動作）。
        """
        node = self.nodes[node_id]
        axis_start = np.asarray(axis_start, dtype=np.float64)
        axis_end = np.asarray(axis_end, dtype=np.float64)
        axis_dir = self._normalize(axis_end - axis_start)
        R = self._rotation_matrix_axis_angle(axis_dir, angle)
        eps = 1e-9

        new_polys = []
        for poly in node.polygons:
            d, _ = self._signed_distance_to_plane(poly, axis_start, axis_end)
            # 片側・分割はここで実装（簡略化：負側を全部回転など）
            if np.any(d < -eps) and np.any(d > eps):
                pos, neg = self._split_polygon_by_line(poly, axis_start, axis_end)
                # 負側を回転
                rot_neg = [self._rotate_points(p, axis_start, R) for p in neg]
                new_polys.extend(pos)
                new_polys.extend(rot_neg)
            elif np.all(d <= eps):
                # 負側 -> 回転
                new_polys.append(self._rotate_points(poly, axis_start, R))
            else:
                # 正側 -> そのまま
                new_polys.append(poly)
        node.polygons = new_polys

    def _rotate_points(self, points, origin, R):
        moved = points - origin
        rotated = moved @ R.T
        return rotated + origin

    # ========== 170度折り（ノード分割あり） ==========

    def apply_soft_fold_to_node(self, node_id, axis_start, axis_end, angle_deg=170.0):
        """
        170度折り：指定ノードを「動く側」と「動かない側」に分け、
        動く側から新しい子ノードを生やす。
        """
        node = self.nodes[node_id]
        axis_start = np.asarray(axis_start, dtype=np.float64)
        axis_end = np.asarray(axis_end, dtype=np.float64)
        axis_dir = self._normalize(axis_end - axis_start)
        angle = np.deg2rad(angle_deg)
        R = self._rotation_matrix_axis_angle(axis_dir, angle)
        eps = 1e-9

        stay_polys = []   # ノードに残る側
        move_polys = []   # 新ノードに移る側

        for poly in node.polygons:
            d, _ = self._signed_distance_to_plane(poly, axis_start, axis_end)
            if np.any(d < -eps) and np.any(d > eps):
                pos, neg = self._split_polygon_by_line(poly, axis_start, axis_end)
                # ここでは例として「負側を動く側」にする
                stay_polys.extend(pos)
                move_polys.extend(neg)
            elif np.all(d <= eps):
                move_polys.append(poly)
            else:
                stay_polys.append(poly)

        # 元ノードは stay_polys のみ残す
        node.polygons = stay_polys

        if len(move_polys) == 0:
            return None  # 分岐なし

        # 動く側を回転
        moved_rotated = [self._rotate_points(p, axis_start, R) for p in move_polys]

        # 新しい法線ベクトル（単純には元の法線をRで回す）
        new_normal = self._rotate_points(node.normal.reshape(1, 3), np.zeros(3), R)[0]

        # 新ノードを生成して子としてぶら下げる
        new_id = self._new_node_id()
        child = SurfaceNode(new_id, moved_rotated, new_normal, parent=node)
        node.children.append(child)
        self.nodes[new_id] = child

        return new_id
