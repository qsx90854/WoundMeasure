"""Bounded center-only localization after an accepted Region-SIFT search.

No reranking or rescue. Invalid rows retain fixed-denominator penalties.
Failure preserves the original accepted coordinate and all original scores.
"""
from dataclasses import replace
import time
import cv2
import numpy as np


def validate_center_config(c):
    if c.center_refine_enabled and c.local_plane_refine:
        raise ValueError('中心精修與 LocalPlane 視差微調不可同時啟用')
    for name in ('points', 'width_px', 'height_px', 'min_points', 'min_quadrants'):
        value = getattr(c, 'center_refine_'+name)
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError('center_refine_'+name+' must be a positive integer')
    if not 3 <= c.center_refine_points <= 64:
        raise ValueError('center_refine_points must be 3..64 (plus P)')
    if c.center_refine_min_points > c.center_refine_points+1 or c.center_refine_min_quadrants > 4:
        raise ValueError('Center minimum evidence exceeds available anchors/quadrants')
    for name in ('max_support_radius_px', 'radius_px', 'step_px', 'max_radius_px', 'spatial_sigma_px', 'max_score'):
        value = getattr(c, 'center_refine_'+name)
        if not np.isfinite(value) or value <= 0:
            raise ValueError('center_refine_'+name+' must be finite and positive')
    step = c.center_refine_step_px
    for radius in (c.center_refine_radius_px, c.center_refine_max_radius_px):
        if radius < step or abs(radius/step-round(radius/step)) > 1e-6:
            raise ValueError('Center radii must be positive integer multiples of step')
    if c.center_refine_radius_px > c.center_refine_max_radius_px or c.center_refine_max_radius_px/step > 32:
        raise ValueError('Center max radius must cover initial radius; at most 65 samples')
    for name in ('min_group_fraction', 'min_weight_fraction', 'min_valid_fraction',
                 'min_relative_gain', 'min_valley_depth', 'max_score_ratio'):
        value = getattr(c, 'center_refine_'+name)
        if not np.isfinite(value) or not 0 < value <= 1:
            raise ValueError('center_refine_'+name+' must be in (0,1]')
    for name in ('min_gain', 'min_point_distance_px', 'competitor_exclusion_px'):
        value = getattr(c, 'center_refine_'+name)
        if not np.isfinite(value) or value < 0:
            raise ValueError('center_refine_'+name+' must be finite and nonnegative')
    if c.center_refine_enabled:
        from Algorithm.region_sift_frames import _scale_specs
        _scale_specs(center_config(c))


def center_config(c):
    n = c.center_refine_points+1
    required = max(c.center_refine_min_points, int(np.ceil(n*c.center_refine_min_group_fraction)))
    return replace(c, center_refine_enabled=False, local_plane_refine=False, two_stage_search=False,
        auto_scale_orientation=True, descriptor_max_support_radius_px=c.center_refine_max_support_radius_px,
        grid_rows=1, grid_cols=1, cell_width_px=c.center_refine_width_px, cell_height_px=c.center_refine_height_px,
        points_per_cell=c.center_refine_points, min_point_distance_px=c.center_refine_min_point_distance_px,
        keep_best_count=required, masked_min_group_fraction=c.center_refine_min_group_fraction,
        masked_min_valid_fraction=c.center_refine_min_valid_fraction, masked_max_deficient_cells=0,
        masked_min_points_per_cell=1)


def sample_center(gray, p, c, mask=None):
    """Spatial strata with alternating high/mid/low gradient targets; P retained."""
    from Algorithm.masked_sift_descriptor import compute_masked_sampling_gradient, exclusion_mask
    p = np.asarray(p, float)
    w, h, count = c.center_refine_width_px, c.center_refine_height_px, c.center_refine_points
    x0, y0 = np.rint(p).astype(int)-np.array([w//2, h//2])
    x1, y1 = x0+w, y0+h
    if x0 < 0 or y0 < 0 or x1 > gray.shape[1] or y1 > gray.shape[0]:
        raise ValueError('center ROI crosses image boundary')
    excluded = np.zeros_like(gray, bool) if mask is None else np.asarray(mask)>0
    if c.use_masked_sift:
        excluded = exclusion_mask(excluded.astype(np.uint8), c) > 0
    px, py = np.rint(p).astype(int)
    if excluded[py, px]:
        raise ValueError('P overlaps excluded pixels')
    if c.use_masked_sift:
        magnitude, clean = compute_masked_sampling_gradient(gray, excluded, (x0,y0,x1,y1), c)
    else:
        magnitude = cv2.magnitude(cv2.Sobel(gray,cv2.CV_32F,1,0,ksize=c.sobel_ksize),
                                 cv2.Sobel(gray,cv2.CV_32F,0,1,ksize=c.sobel_ksize))
        clean = ~excluded
    cols = int(np.ceil(np.sqrt(count))); rows = int(np.ceil(count/cols))
    if w < cols or h < rows:
        raise ValueError('center ROI too small for stratified anchors')
    xs, ys = np.linspace(x0,x1,cols+1,dtype=int), np.linspace(y0,y1,rows+1,dtype=int)
    strata = np.rint(np.linspace(0, rows*cols-1, count)).astype(int)
    selected = []
    for j, cell in enumerate(strata):
        r, col = divmod(int(cell), cols)
        yy, xx = np.mgrid[ys[r]:ys[r+1],xs[col]:xs[col+1]]
        coords = np.column_stack((xx.ravel(),yy.ravel()))
        coords = coords[clean[coords[:,1],coords[:,0]]]
        if not len(coords):
            raise ValueError('center stratum has no unmasked pixels')
        order = np.argsort(magnitude[coords[:,1],coords[:,0]],kind='stable')
        rank = int(round((.9,.5,.1)[j%3]*(len(order)-1)))
        chosen = None
        for k in sorted(range(len(order)),key=lambda i:(abs(i-rank),i)):
            pt = coords[order[k]].astype(float)
            distances = np.linalg.norm(np.asarray([*selected, p])-pt,axis=1)
            if np.any(distances < max(1e-6,c.center_refine_min_point_distance_px)):
                continue
            chosen=pt; break
        if chosen is None:
            raise ValueError('center stratum cannot satisfy point spacing')
        selected.append(chosen)
    return np.asarray([*selected,p],np.float32), (int(x0),int(y0),w,h)


def weighted_curve(distances, valid, points, p, c):
    """Fixed weights/denominator; missing pairs are penalized, never dropped."""
    relative = np.asarray(points)-p
    weights = np.exp(-np.sum(relative**2,axis=1)/(2*c.center_refine_spatial_sigma_px**2))
    weights /= weights.sum()
    ceiling = np.sqrt(2.)*(1. if c.normalize_descriptors else 512.)
    scores = np.sum(np.where(valid, distances, ceiling)*weights,axis=1)
    required = max(c.center_refine_min_points,int(np.ceil(len(points)*c.center_refine_min_group_fraction)))
    usable = (valid.sum(axis=1)>=required)&((valid*weights).sum(axis=1)>=c.center_refine_min_weight_fraction)
    for i in np.flatnonzero(usable):
        xy = relative[valid[i]]
        off_center = np.linalg.norm(xy,axis=1)>1e-6
        q = (xy[off_center,0]>=0).astype(int)+2*(xy[off_center,1]>=0).astype(int)
        hull = cv2.convexHull(xy.astype(np.float32))
        usable[i] = len(np.unique(q))>=c.center_refine_min_quadrants and cv2.pointPolygonTest(hull,(0.,0.),False)>0
    return np.where(usable,scores,np.nan), weights


def assess_curve(offsets, scores, c):
    x, y = np.asarray(offsets), np.asarray(scores)
    z = int(np.argmin(abs(x)))
    if not np.isfinite(y[z]): return dict(action='fallback',reason='original center score lacks evidence')
    ids=np.flatnonzero(np.isfinite(y))
    if len(ids)<3: return dict(action='fallback',reason='insufficient valid local samples')
    best=int(ids[np.argmin(y[ids])]); gain=float(y[z]-y[best])
    info=dict(index=best,shift=float(x[best]),original_score=float(y[z]),best_score=float(y[best]),gain=gain)
    if best==z: return dict(info,action='fallback',reason='original center already best')
    if gain < c.center_refine_min_gain or gain/max(abs(y[z]),1e-8)<c.center_refine_min_relative_gain:
        return dict(info,action='fallback',reason='center score improvement too small')
    if not np.all(np.isfinite(y[min(z,best):max(z,best)+1])):
        return dict(info,action='fallback',reason='invalid samples between original and minimum')
    if y[best]>c.center_refine_max_score: return dict(info,action='fallback',reason='center score above limit')
    if best in (0,len(x)-1):
        neighbor=1 if best==0 else len(x)-2
        if np.isfinite(y[neighbor]) and y[best]<y[neighbor]:
            return dict(info,action='expand',reason='improving minimum at boundary')
        return dict(info,action='fallback',reason='uncertain boundary minimum')
    if not np.all(np.isfinite(y)):
        return dict(info,action='fallback',reason='local curve contains unsupported gaps')
    shoulder=min(y[:best].max(), y[best+1:].max())
    depth=(shoulder-y[best])/max(abs(shoulder),1e-8)
    rivals=np.abs(x-x[best])>c.center_refine_competitor_exclusion_px
    rival=float(np.min(y[rivals])) if np.any(rivals) else np.nan
    ratio=float(y[best]/max(rival,1e-8)) if np.isfinite(rival) else np.nan
    info.update(relative_depth=float(depth),competitor_ratio=ratio)
    if depth<c.center_refine_min_valley_depth or not np.isfinite(ratio) or ratio>c.center_refine_max_score_ratio:
        return dict(info,action='fallback',reason='center minimum is shallow or ambiguous')
    return dict(info,action='apply',reason='distinct center minimum')


def triangulate_checked(point, right, cand, K):
    a=np.asarray(point,np.float64).reshape(1,1,2); b=np.asarray(right,np.float64).reshape(1,1,2)
    if cand.get('F') is not None: a,b=cv2.correctMatches(cand['F'],a,b)
    R=np.asarray(cand['R_rel']); t=np.asarray(cand['t_rel']).reshape(3,1)
    P0=np.asarray(K)@np.column_stack((np.eye(3),np.zeros(3)))
    P1=np.asarray(cand['K_R'])@np.column_stack((R,t))
    v=cv2.triangulatePoints(P0.astype(np.float32),P1.astype(np.float32),
                           a.reshape(2,1).astype(np.float32),b.reshape(2,1).astype(np.float32)).reshape(4)
    if abs(v[3])<1e-10: raise ValueError('center triangulation at infinity')
    xyz=v[:3]/v[3]
    if not np.all(np.isfinite(xyz)) or xyz[2]<=0 or (R@xyz+t.ravel())[2]<=0:
        raise ValueError('center triangulation behind camera or non-finite')
    return xyz


def refine_center(result, left, point, cand, K, config, *, left_spec_mask=None):
    if not config.center_refine_enabled: return result
    started=time.perf_counter()
    report=dict(status='skipped',applied=False,reason='original match rejected',
                descriptor_rows=0,descriptor_batches=0,left_descriptor_rows=0,elapsed_ms=0.)
    result['center_refinement']=report
    debug=result.get('region_debug')
    if debug is not None: debug['center_refinement']=report
    counts=dict(right_descriptor_rows=0,right_descriptor_batches=0)
    try:
        validate_center_config(config)
        if result.get('m_pt') is None or result.get('reject_reason') or debug is None: return result
        from Algorithm import Region_SIFT_Matching as matcher
        from Algorithm import masked_sift_descriptor as masked
        from Algorithm.region_sift_search import frame_context, evaluate_positions
        original=np.asarray(result['m_pt']).copy()
        report.update(status='fallback',reason='center evidence unavailable',original_center_right=original,
                      original_method=result.get('method',''),original_center_warp=debug['best_center_warp'].copy())
        cfg=center_config(config); p=np.asarray(point,np.float32)
        mask=left_spec_mask if debug['reject_specular'] else None
        if cfg.use_masked_sift and mask is None: mask=np.zeros_like(left,np.uint8)
        points,roi=sample_center(left,p,config,mask)
        metadata=[dict(index=i,is_click=i==len(points)-1,cell_row=0,cell_col=0) for i in range(len(points))]
        ctx=frame_context(left,cfg,{},'left',mask,roi=matcher._left_context_roi(cfg,roi,left.shape))
        origin=np.asarray(ctx.get('origin',(0,0)),np.float32); ox,oy=origin.astype(int); ch,cw=ctx['image_shape']
        support_mask=None
        if debug['reject_specular'] and cfg.specular_check_support and not cfg.use_masked_sift:
            support_mask=np.where(mask>0,0,255).astype(np.uint8)[oy:oy+ch,ox:ox+cw]
        lf=matcher.estimate_dense_sift_frames(left[oy:oy+ch,ox:ox+cw],points-origin,
            replace(cfg,min_valid_warp_ratio=1.) if support_mask is not None else cfg,context=ctx,valid_mask=support_mask)
        if not np.all(lf['valid']): raise ValueError('center anchors lack image/support validity')
        packet=None
        if cfg.use_masked_sift:
            packet=masked.compute_descriptors(ctx,points-origin,lf['size_px'],lf['angle_deg'],cfg)
            ld=packet['descriptors']
        else:
            ld=matcher.compute_descriptors_at_points(left,points,config=cfg,
                sizes_px=lf['size_px'],angles_deg=lf['angle_deg'],octaves=lf['octave'])
        report['left_descriptor_rows']=len(points)
        if not np.any(np.linalg.norm(ld,axis=1)>1e-8): raise ValueError('all center descriptors are uninformative')
        base=np.asarray(debug['best_center_warp']); tangent=np.asarray(debug['tangent'])
        image=debug['warped_right_gray']; valid=debug['warped_valid_mask']; spec=debug['warped_specular_mask']
        session,cache={},{}
        along=debug['best_along_offset_px']; across=debug['best_across_offset_px']
        step=config.center_refine_step_px; radius=config.center_refine_radius_px
        maxradius=config.center_refine_max_radius_px
        band=dict(seed_on_line=debug['seed_on_line'],tangent=tangent,normal=debug['normal'])
        right_roi=matcher._right_context_roi(cfg,band,points-p,image.shape)
        report.update(points=points,weights=None,roi=roi,history=[],actual_left_support_max_px=float(max(lf['support_radius_px'])))
        while True:
            offsets=np.arange(-round(radius/step),round(radius/step)+1)*step
            centers=base+offsets[:,None]*tangent
            positions=(centers[:,None]+(points-p)[None]).astype(np.float32)
            eligible=np.flatnonzero((abs(along+offsets)<=(cfg.search_length_px-1)/2+1e-6)
                &np.all((positions[:,:,0]>=0)&(positions[:,:,0]<=image.shape[1]-1)
                        &(positions[:,:,1]>=0)&(positions[:,:,1]<=image.shape[0]-1),axis=1))
            if not len(eligible): raise ValueError('center search outside original band/image')
            # As in the main matcher, require valid anchor centers before frames.
            rounded=np.rint(positions[eligible]).astype(int)
            good=np.all(valid[rounded[:,:,1],rounded[:,:,0]]>0,axis=1)
            if debug['reject_specular'] and not cfg.use_masked_sift and spec is not None:
                good &= np.all(spec[rounded[:,:,1],rounded[:,:,0]]==0,axis=1)
            eligible=eligible[good]
            output=evaluate_positions(image,positions,eligible,valid,spec,debug['reject_specular'],
                dict(centers_warp=centers,across_offsets=np.full(len(offsets),across)),ld,metadata,lf,None,cfg,
                session,cache,lambda stage:None,counts,left_masked=packet,right_context_roi=right_roi)
            distances=np.full((len(offsets),len(points)),np.nan)
            pair_ok=np.zeros_like(distances,bool)
            for i,row in output[8].items():
                distances[i]=output[5][i]
                pair_ok[i]=output[7].get('masked_point_valid',np.ones((len(output[8]),len(points)),bool))[row]
            scores,weights=weighted_curve(distances,pair_ok,points,p,config)
            decision=assess_curve(offsets,scores,config)
            report.update(offsets=offsets,scores=scores,valid_counts=pair_ok.sum(axis=1),weights=weights,
                          distances=distances,point_costs=np.where(pair_ok,distances,np.nan),
                          reason=decision['reason'],decision=decision)
            report['history'].append(dict(radius=radius,action=decision['action'],reason=decision['reason']))
            if decision['action']=='expand' and radius<maxradius:
                radius=min(maxradius,radius+config.center_refine_radius_px); continue
            if decision['action']!='apply':
                if decision['action']=='expand': report['reason']='minimum at maximum search radius; original retained'
                return result
            center=base+decision['shift']*tangent
            right=matcher._transform_points(center.reshape(1,2),debug['H_lr'])[0]
            old_xyz=triangulate_checked(point,original,cand,K)
            new_xyz=triangulate_checked(point,right,cand,K)
            report.update(status='applied',applied=True,reason='center score and geometry verified',
                center_shift_px=decision['shift'],refined_center_warp=center,refined_center_right=right,
                original_p3d=old_xyz,refined_p3d=new_xyz)
            result['m_pt']=right.astype(np.float32)
            result['method']+='+CenterRefine'
            return result
    except (ValueError,KeyError,TypeError,RuntimeError,cv2.error,np.linalg.LinAlgError) as exc:
        report['reason']='center fallback: '+str(exc)
        return result
    finally:
        report.update(elapsed_ms=(time.perf_counter()-started)*1000,
            descriptor_rows=counts['right_descriptor_rows'],descriptor_batches=counts['right_descriptor_batches'])
        for key in ('right_descriptor_rows','right_descriptor_batches'):
            result.setdefault('timing_counts',{})[key]=result.get('timing_counts',{}).get(key,0)+counts[key]


def attach_center_heights(result, triangulate, normal, center, *, offset=0., max_depth=np.inf):
    """MAIN's exact geometry/plane; invalid final geometry falls back, never rescues."""
    r=result.get('center_refinement')
    if not r or r['status']=='skipped': return
    if result.get('m_pt') is None or result.get('reject_reason'):
        r.update(status='discarded', applied=False, reason='upstream app gate rejected: '+str(result.get('reject_reason')),
                 final_height_mm=None)
        return
    try:
        old=triangulate(r['original_center_right'])
        new=triangulate(result['m_pt']) if result.get('m_pt') is not None else old
        if new is None or not np.all(np.isfinite(new)) or not 0<new[2]<=max_depth:
            raise ValueError('refined geometry exceeds app depth limits')
        if normal is not None and center is not None:
            r['original_height_mm']=float(np.dot(normal,old-center))-offset
            r['final_height_mm']=float(np.dot(normal,new-center))-offset
    except (ValueError,TypeError,cv2.error,FloatingPointError) as exc:
        if r.get('applied') and result.get('m_pt') is not None:
            result['m_pt']=r['original_center_right'].copy(); result['method']=r['original_method']
            r.update(status='fallback',applied=False,reason='app geometry fallback: '+str(exc))
            r['final_height_mm']=r.get('original_height_mm')


def summary_for_log(report):
    if not report: return None
    keys=('status','applied','reason','center_shift_px','original_center_right','refined_center_right',
          'original_height_mm','final_height_mm','left_descriptor_rows','descriptor_rows','descriptor_batches',
          'elapsed_ms','actual_left_support_max_px','decision','history')
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in report.items() if k in keys}


class CenterRefinementViewer:
    def __init__(self): self.figure=None
    def close(self):
        if self.figure is not None:
            import matplotlib.pyplot as plt
            plt.close(self.figure); self.figure=None
    def show(self,result,enabled=True):
        r=(result.get('region_debug') or {}).get('center_refinement')
        if not enabled or not r or 'offsets' not in r:
            self.close(); return
        import matplotlib.pyplot as plt
        if self.figure is None or not plt.fignum_exists(self.figure.number):
            self.figure=plt.figure(figsize=(10,6))
            self.figure.canvas.manager.set_window_title('Region-SIFT center refinement')
        f=self.figure; f.clear(); ax=f.add_subplot(111)
        ax.plot(r['offsets'],r['scores'],'o-',label='Center weighted score (NOT original Objective)')
        ax.axvline(0,color='gray',ls='--',label='Original match')
        if r.get('applied'): ax.axvline(r['center_shift_px'],color='green',label='Applied center correction')
        for x,y,n in zip(r['offsets'],r['scores'],r['valid_counts']):
            if np.isfinite(y): ax.annotate(str(n),(x,y),xytext=(0,8),textcoords='offset points',ha='center')
        ax.set(xlabel='Along correction from original Top 1 (px)',ylabel='Center score (lower is better)',
               title=f'{r["status"]}: {r["reason"]}\nValid point counts above samples; gaps = insufficient evidence')
        ax.grid(alpha=.2); ax.legend(); f.tight_layout(); f.canvas.draw_idle(); f.show()
